import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

class FlutterPluginTensorflow {
  static const MethodChannel _channel =
      const MethodChannel('flutter_plugin_tensorflow');

  static Future<String> get platformVersion async {
    final String version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }

  ///加载模型 key 用来标记 tf
  static Future<bool> loadMode(String key,String assetPath)async{
    return await _channel.invokeMethod('loadMode',{"key":key,"path":assetPath});
  }
  ///加载帧,判断是否是手  0.8 认为是手
  static Future<bool>runPalmModelOnFrame(String key, List<Uint8List> bytesList,{int inputW = 224,int  inputH =224,int rotation =90,double separator =0.8})async{
    return await _channel.invokeMethod('runPalmModelOnFrame',{"key":key,"data":bytesList,
    "width":inputW,"height":inputH,"separator":separator,"rotation":rotation});
  }
  ///加载帧,得到画线图片
  static Future<Uint8List>runPalmPrintModelOnFrame(String key, List<Uint8List> bytesList,{int inputW = 312,int  inputH =312,int rotation =90,double separator =0.8})async{
    return await _channel.invokeMethod('runPalmPrintModelOnFrame',{"key":key,"data":bytesList,
      "width":inputW,"height":inputH,"separator":separator,"rotation":rotation});
  }
  ///关闭模型
  static Future<bool>close()async{
    return await _channel.invokeMethod('close');
  }
}
