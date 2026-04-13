#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface LibRawBridge : NSObject
+ (nullable CGImageRef)decodeFileAtPath:(NSString *)path CF_RETURNS_RETAINED;
@end
