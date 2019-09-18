package com.fye.flutter.plugin.flutter_plugin_tensorflow

import android.content.Context
import android.os.AsyncTask
import android.renderscript.*
import android.util.Log
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry.Registrar
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY
import java.util.*
import kotlin.collections.HashMap
import jdk.nashorn.internal.objects.NativeArray.forEach
import android.renderscript.Element.U8
import android.renderscript.Element.U8_4
import android.graphics.Bitmap
import java.nio.ByteOrder.nativeOrder
import android.R.attr.shape
import android.graphics.Canvas
import org.tensorflow.lite.Tensor
import java.nio.ByteOrder


class FlutterPluginTensorflowPlugin : MethodCallHandler {

    companion object {

        private lateinit var mRegistrar: Registrar
        //用于存放 Interpreter
        private val map = HashMap<String, Interpreter>()
        private var tfLiteBusy = false
        @JvmStatic
        fun registerWith(registrar: Registrar) {
            mRegistrar = registrar
            val channel = MethodChannel(registrar.messenger(), "flutter_plugin_tensorflow")
            channel.setMethodCallHandler(FlutterPluginTensorflowPlugin())
        }
    }


    override fun onMethodCall(call: MethodCall, result: Result) {
        when {
            call.method == "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
            call.method == "loadMode" -> {
                try {
                    val res = loadModel(call.arguments as HashMap<String, String>)
                    result.success(res)
                } catch (e: Exception) {
                    result.error("Failed to load model", e.message, e)
                }

            }
            call.method == "runPalmModelOnFrame" -> {
                try {
                    var map = call.arguments as HashMap<String, Objects>
                    PalmTfliteTask(map, result).executeTfliteTask()
                } catch (e: Exception) {
                    result.error("Failed to load model", e.message, e)
                }
            }
            call.method == "runPalmPrintModelOnFrame" -> {

            }
            call.method == "close" -> {

            }
            else -> result.notImplemented()
        }
    }

    @Throws(IOException::class)
    private fun loadModel(args: HashMap<String, String>): Boolean {
        val model = args["path"].toString()
        val name = args["key"].toString()
        val assetManager = mRegistrar.context().assets
        var key = mRegistrar.lookupKeyForAsset(model)
        val fileDescriptor = assetManager.openFd(key)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val buffer = fileChannel.map(READ_ONLY, startOffset, declaredLength)
        val tfliteOptions = Interpreter.Options()
        tfliteOptions.setNumThreads(3)
        tfliteOptions.addDelegate(GpuDelegate())
        var tfLite = Interpreter(buffer, tfliteOptions)
        map[name] = tfLite

        return true
    }

    fun renderScriptNV21ToRGBA888(context: Context, width: Int, height: Int, nv21: ByteArray): Allocation {
        // https://stackoverflow.com/a/36409748
        val rs = RenderScript.create(context)
        val yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, U8_4(rs))
        val yuvType = Type.Builder(rs, U8(rs)).setX(nv21.size)
        val input = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)

        val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height)
        val out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)

        input.copyFrom(nv21)

        yuvToRgbIntrinsic.setInput(input)
        yuvToRgbIntrinsic.forEach(out)
        return out
    }
    @Throws(IOException::class)
    fun feedInputTensor(bitmapRaw: Bitmap, mean: Float, std: Float): ByteBuffer {
        val tensor = tfLite.getInputTensor(0)
        val shape = tensor.shape()
        inputSize = shape[1]
        val inputChannels = shape[3]

        val bytePerChannel = if (tensor.dataType() == DataType.UINT8) 1 else BYTES_PER_CHANNEL
        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel)
        imgData.order(ByteOrder.nativeOrder())

        var bitmap = bitmapRaw
        if (bitmapRaw.width != inputSize || bitmapRaw.height != inputSize) {
            val matrix = getTransformationMatrix(bitmapRaw.width, bitmapRaw.height,
                    inputSize, inputSize, false)
            bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawBitmap(bitmapRaw, matrix, null)
        }

        if (tensor.dataType() == DataType.FLOAT32) {
            for (i in 0 until inputSize) {
                for (j in 0 until inputSize) {
                    val pixelValue = bitmap.getPixel(j, i)
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - mean) / std)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - mean) / std)
                    imgData.putFloat(((pixelValue and 0xFF) - mean) / std)
                }
            }
        } else {
            for (i in 0 until inputSize) {
                for (j in 0 until inputSize) {
                    val pixelValue = bitmap.getPixel(j, i)
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                }
            }
        }

        return imgData
    }
    //任务抽象类
    private abstract class TfliteTask constructor(map: HashMap<String, Objects>, result: Result) : AsyncTask<Void, Void, Void>() {
        init {
            if (tfLiteBusy) throw  RuntimeException("Interpreter busy")
            else tfLiteBusy = true
        }

        abstract fun runTflite()

        abstract fun onRunTfliteDone()

        override fun doInBackground(vararg backgroundArguments: Void): Void? {
            runTflite()
            return null
        }

        override fun onPostExecute(backgroundResult: Void) {
            tfLiteBusy = false
            onRunTfliteDone()
        }

        fun executeTfliteTask() {
            runTflite()
            tfLiteBusy = false
            onRunTfliteDone()
        }
    }

    //识别手相 任务
    private class PalmTfliteTask(map: HashMap<String, Objects>, result: Result) : TfliteTask(map, result) {
        lateinit var input: ByteBuffer
        var key: String = map["key"] as String
        var data: List<ByteArray> = map["data"] as List<ByteArray>
        var width: Int = map["width"] as Int
        var height: Int = map["height"] as Int
        var separator: Double = map["separator"] as Double
        var rotation: Int = map["rotation"] as Int
        var result: Result = result

        var output = Array(1) { FloatArray(2) }
        override fun runTflite() {
            map[key]?.run(input, output)
        }

        override fun onRunTfliteDone() {
            val background = output[0][0]
            val palm = output[0][1]
            log("检测到背景概率:$background,  检测到手概率:$palm")
            result.success(false)
        }

    }
}

fun log(str: String) {
    Log.i("===>TensorFlow", "$str")
}
