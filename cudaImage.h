/**
 *  Author: Marten Bjorkman
 *  Contributor: Ethan Stewart
 *
 *  This file consists of a class CudaImage and some functions
 *  which may pad the allocation to ensure that corresponding
 *  pointers in any given row will continue to meet the alignment
 *  requirements for coalescing as the address is updated from row to row.
 */

#ifndef CUDASIFT_CUDAIMAGE_H
#define CUDASIFT_CUDAIMAGE_H

/**
 * The CudaImage defines a class that performs memory allocation
 * and data transfer between host and device
 * @Param width Requested pitched allocation width(in bytes)
 * @Param height Requested pitched allocation height
 * @Param pitch Pitch for allocation
 * @param h_data Pointer to allocated host memory
 * @param d_data Pointer to allocated pitched device memory
 * @param t_data Pointer to allocated texture memory
 * @Param d_internalAlloc Boolean value indicating whether to call the Allocate method to allocate memory on the device
 * @Param h_internalAlloc Boolean value indicating whether to call the Allocate method to allocate memory on the host
 */
class CudaImage {

public:
    int width, height, pitch;
    float *h_data;
    float *d_data;
    float *t_data;
    bool d_internalAlloc;
    bool h_internalAlloc;

public:
    CudaImage();

    ~CudaImage();

    /**
     * Allocate host memory and device memory.
     * Parameter withHost indicates whether to allocate host memory.
     * Destructor will free space for devMem and hostMem.
     * Both devMem and hostMem could be NULL.
     */
    void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);

    /**
     * Copy data from host to device.Pay attention to width,
     * height, pitch, once not corrected, it may break down.
     * @return Time for coping data.
     */
    double Download();

    /**
     * Copy data from device to host.
     * @return Time for coping data.
     */
    double Readback();

    double InitTexture();

    /**
     * Copy data from device to host.
     * @return Time for coping data.
     */
    double CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int dividend, int divisor);

int iDivDown(int dividend, int divisor);

int iAlignUp(int number, int alignment);

int iAlignDown(int number, int alignment);

void StartTimer(unsigned int *hTimer);

double StopTimer(unsigned int hTimer);

#endif // CUDASIFT_CUDAIMAGE_H