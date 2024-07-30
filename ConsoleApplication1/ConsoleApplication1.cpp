#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>

using namespace std;

void grad_x(float* output, float* input, int width, int height) {
    assert(output != NULL && input != NULL && width > 0 && height > 0);

    int w1 = width - 1;

    int i, j;
    for (i = 0;i < height;++i) {
        j = 0;
        output[j + i * width] = input[j + 1 + i * width] - input[j + i * width];
        for (j = 1;j < w1;++j) {
            output[j + i * width] = 0.5f * (input[j + 1 + i * width] - input[j - 1 + i * width]);
        }
        output[j + i * width] = input[j + i * width] - input[j - 1 + i * width];
    }
}

void grad_y(float* output, float* input, int width, int height)
{
    assert(output != NULL && input != NULL && width > 0 && height > 0);

    int h1 = height - 1;

    int i, j;

    i = 0;
    for (j = 0;j < width;++j)
    {
        output[j + i * width] = input[j + (i + 1) * width] - input[j + i * width];
    }
    for (i = 1;i < h1;++i)
    {
        for (j = 0;j < width;++j)
        {
            output[j + i * width] = 0.5f * (input[j + (i + 1) * width] - input[j + (i - 1) * width]);
        }
    }
    for (j = 0;j < width;++j)
    {
        output[j + i * width] = input[j + i * width] - input[j + (i - 1) * width];
    }
}

#include <omp.h>
#include <assert.h>

void your_grad_x(float* output, float* input, int width, int height) {
    assert(output != NULL && input != NULL && width > 0 && height > 0);


    #pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        float* out = &output[i * width];
        register float* in = &input[i * width * 2];

        out[0] = in[0] - in[1];

        //TODO
        //alignment_malloc을 32바이트(4*8)단위로 output, input을 초기화하며 인덱스 1~1912를 넣어야함 0~1919, 0~3838로 초기화하면 8개씩 끊어서 언롤할때 안맞음
        //j loop에 parallel?
        register int j;
        for (j = 1; j < 1913; j += 8) {
            out[j] = 0.5f * (in[j * 2] - in[j * 2 + 1]);
            out[j + 1] = 0.5f * (in[j * 2 + 2] - in[j * 2 + 3]);
            out[j + 2] = 0.5f * (in[j * 2 + 4] - in[j * 2 + 5]);
            out[j + 3] = 0.5f * (in[j * 2 + 6] - in[j * 2 + 7]);
            out[j + 4] = 0.5f * (in[j * 2 + 8] - in[j * 2 + 9]);
            out[j + 5] = 0.5f * (in[j * 2 + 10] - in[j * 2 + 11]);
            out[j + 6] = 0.5f * (in[j * 2 + 12] - in[j * 2 + 13]);
            out[j + 7] = 0.5f * (in[j * 2 + 14] - in[j * 2 + 15]);
        }
        //printf("j=%d\n", j);
        out[j] = 0.5f * (in[j * 2] - in[j * 2 + 1]);
        out[j + 1] = 0.5f * (in[(j + 1) * 2] - in[(j + 1) * 2 + 1]);
        out[j + 2] = 0.5f * (in[(j + 2) * 2] - in[(j + 2) * 2 + 1]);
        out[j + 3] = 0.5f * (in[(j + 3) * 2] - in[(j + 3) * 2 + 1]);
        out[j + 4] = 0.5f * (in[(j + 4) * 2] - in[(j + 4) * 2 + 1]);
        out[j + 5] = 0.5f * (in[(j + 5) * 2] - in[(j + 5) * 2 + 1]);
        out[j + 6] = in[(j + 6) * 2] - in[(j + 6) * 2 + 1];
    }
}



void your_grad_y(float* output, float* input, int width, int height)
{
    assert(output != NULL && input != NULL && width > 0 && height > 0);

    int h1 = height - 1;

    int i, j;

    i = 0;
    for (j = 0;j < width;++j)
    {
        output[j + i * width] = input[j + (i + 1) * width] - input[j + i * width];
    }
    for (i = 1;i < h1;++i)
    {
        for (j = 0;j < width;++j)
        {
            output[j + i * width] = 0.5f * (input[j + (i + 1) * width] - input[j + (i - 1) * width]);
        }
    }
    for (j = 0;j < width;++j)
    {
        output[j + i * width] = input[j + i * width] - input[j + (i - 1) * width];
    }
}


int main()
{
    omp_set_num_threads(4);
    // load the image
    int width = 1920; // width of the image
    int height = 1080;  // height of the image
    int len = width * height; // pixels in the image

    float* data = (float*)malloc(sizeof(float) * len);  // image buffer
    //float* data_rearranged = (float*)malloc(sizeof(float) * len * 2);  // image buffer rearrange
    float* data_rearranged = (float*)_aligned_malloc(sizeof(float) * len * 2, 32);

    float* gx = (float*)malloc(sizeof(float) * len);  // original output buffer
    float* gy = (float*)malloc(sizeof(float) * len);  // original output buffer
    //float* your_gx = (float*)malloc(sizeof(float) * len); // your output buffer
    //float* your_gy = (float*)malloc(sizeof(float) * len); // your output buffer
    float* your_gx = (float*)_aligned_malloc(sizeof(float) * len, 32);
    float* your_gy = (float*)_aligned_malloc(sizeof(float) * len, 32);

    FILE* fp = fopen("../image.dat", "rb"); // open the image. if this code does not work, change the path.
    fread(data, sizeof(float), width * height, fp); // load the float values, the image is gray.
    fclose(fp);

    for (int i = 0; i < height; ++i) {
        data_rearranged[i * width * 2] = data[i * width + 1];
        data_rearranged[i * width * 2 + 1] = data[i * width];
        
        int j = 1;
        for (; j < width - 1; ++j) {
            data_rearranged[i * width * 2 + (j * 2)] = data[i * width + j + 1];
            data_rearranged[i * width * 2+ (j * 2) + 1] = data[i * width + j - 1];
        }

        data_rearranged[i * width * 2 + (j * 2)] = data[i * width + j];
        data_rearranged[i * width * 2 + (j * 2) + 1] = data[i * width + j - 1];

    }

    // perform gradient computation
    LARGE_INTEGER Frequency;
    LARGE_INTEGER BeginTime;
    LARGE_INTEGER EndTime;

    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&BeginTime);
    grad_x(gx, data, width, height);
    grad_y(gy, data, width, height);
    QueryPerformanceCounter(&EndTime);

    __int64 elapsed = EndTime.QuadPart - BeginTime.QuadPart;
    double duration = (double)elapsed / (double)Frequency.QuadPart;
    printf("naive gradient computation: %2.3f seconds\n", duration);

    QueryPerformanceCounter(&BeginTime);
    your_grad_x(your_gx, data_rearranged, width, height);
    your_grad_y(your_gy, data, width, height);
    QueryPerformanceCounter(&EndTime);

    elapsed = EndTime.QuadPart - BeginTime.QuadPart;
    double duration2 = (double)elapsed / (double)Frequency.QuadPart;
    printf("your gradient computation: %2.3f seconds\n", duration2);

    double diff_x_sum = 0;
    double diff_y_sum = 0;
    for (int i = 0;i < len; ++i) {
        diff_x_sum += fabs(your_gx[i] - gx[i]);
        diff_y_sum += fabs(your_gy[i] - gy[i]);
    }
    printf("diff_x_sum:%lf\ndiff_y_sum:%lf\n", diff_x_sum, diff_y_sum);

    printf("The performance of your gradient computation is %2.3f times higher than naive gradient computation.\n", duration / duration2);

    free(data);
    _aligned_free(data_rearranged);
    free(gx);
    free(gy);
    _aligned_free(your_gx);
    _aligned_free(your_gy);
    getchar();

    return 0;
}