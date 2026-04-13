// Interface targeting LibRaw 0.21+
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface LibRawBridge : NSObject
/// Returns decoded CGImage or nil on failure.
/// Tries embedded JPEG preview first (fast path), falls back to full LibRaw decode.
+ (nullable CGImageRef)decodeFileAtPath:(NSString *)path CF_RETURNS_RETAINED;
@end
