#import "FlutterPluginTensorflowPlugin.h"
#import <flutter_plugin_tensorflow/flutter_plugin_tensorflow-Swift.h>

@implementation FlutterPluginTensorflowPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftFlutterPluginTensorflowPlugin registerWithRegistrar:registrar];
}
@end
