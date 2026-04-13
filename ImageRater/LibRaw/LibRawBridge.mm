#import "LibRawBridge.h"
#import <libraw/libraw.h>

@implementation LibRawBridge

+ (CGImageRef)decodeFileAtPath:(NSString *)path {
    LibRaw processor;
    int ret = processor.open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) return nil;

    // Fast path: try embedded JPEG thumbnail
    if (processor.unpack_thumb() == LIBRAW_SUCCESS) {
        libraw_processed_image_t *thumb = processor.dcraw_make_mem_thumb();
        if (thumb && thumb->type == LIBRAW_IMAGE_JPEG) {
            NSData *jpegData = [NSData dataWithBytes:thumb->data length:thumb->data_size];
            LibRaw::dcraw_clear_mem(thumb);
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)jpegData);
            CGImageRef img = CGImageCreateWithJPEGDataProvider(provider, nil, true, kCGRenderingIntentDefault);
            CGDataProviderRelease(provider);
            processor.recycle();
            return img;
        }
        if (thumb) LibRaw::dcraw_clear_mem(thumb);
    }

    // Full decode fallback
    processor.recycle();
    ret = processor.open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) return nil;
    if (processor.unpack() != LIBRAW_SUCCESS) return nil;
    if (processor.dcraw_process() != LIBRAW_SUCCESS) return nil;

    libraw_processed_image_t *image = processor.dcraw_make_mem_image();
    if (!image) return nil;

    int width = image->width, height = image->height;
    NSData *data = [NSData dataWithBytes:image->data length:image->data_size];
    LibRaw::dcraw_clear_mem(image);
    processor.recycle();

    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef img = CGImageCreate(width, height, 8, 24, width * 3, space,
                                   kCGBitmapByteOrderDefault | kCGImageAlphaNone,
                                   provider, nil, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    return img;
}

@end
