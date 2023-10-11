//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utility.h"
#include "cudaImage.h"
#include "cudaSift.h"

using namespace std;

int
ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    cv::Mat original, verified, unverified;
    std::string receiptType = "../data/mountain ";
    std::string path2origin = receiptType + "original.jpg";
    std::string path2verified = receiptType + "verified.jpg";
    std::string path2unverified = receiptType + "unverified.jpg";
    cv::imread(path2origin, cv::IMREAD_GRAYSCALE).convertTo(original, CV_32FC1);
    cv::imread(path2verified, cv::IMREAD_GRAYSCALE).convertTo(verified, CV_32FC1);
    cv::imread(path2unverified, cv::IMREAD_GRAYSCALE).convertTo(unverified, CV_32FC1);

    unsigned int w = unverified.cols;
    unsigned int h = unverified.rows;

    CudaImage oriCuda, veriCuda, unveCuda;
    oriCuda.Allocate(original.cols, original.rows, iAlignUp(original.cols, 128), false, NULL,
                     (float *) original.data);
    veriCuda.Allocate(verified.cols, verified.rows, iAlignUp(verified.cols, 128), false, NULL,
                      (float *) verified.data);
    unveCuda.Allocate(unverified.cols, unverified.rows, iAlignUp(unverified.cols, 128), false, NULL,
                      (float *) unverified.data);
//    oriCuda.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *) original.data);
//    veriCuda.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *) verified.data);
//    unveCuda.Allocate(w, h, iAlignUp(w, 128), false, NULL,
//                      (float *) unverified.data);
    oriCuda.Download();
    veriCuda.Download();
    unveCuda.Download();

    // Extract Sift features from images
    SiftData siftData1, siftData2, siftData3;

    InitSiftData(siftData1, 32768, true, true);
    InitSiftData(siftData2, 32768, true, true);
    InitSiftData(siftData3, 32768, true, true);

    // A bit of benchmarking
    //for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
    float *memoryTmp = AllocSiftTempMemory(w, h, 7, false);
    float homography[9];
    int numMatches;
    int numFit;
    int numFitfake;
    int numMatchesfake;
    ofstream file("data.txt", ios::app);
    for (int nOctaveLayers = 2; nOctaveLayers <= 6; nOctaveLayers++) {
        for (float thresh = 3.2f; thresh > 0.38f;) {
            for (float edgeThreshold = 18.0f; edgeThreshold > 5.8f;) {
                for (float initBlur = 3.0f; initBlur > 0.88f;) {
                    ExtractSift(siftData1, oriCuda, nOctaveLayers, initBlur, thresh, edgeThreshold, 0.0f, false,
                                memoryTmp);
                    ExtractSift(siftData2, veriCuda, nOctaveLayers, initBlur, thresh, edgeThreshold, 0.0f, false,
                                memoryTmp);
                    ExtractSift(siftData3, unveCuda, nOctaveLayers, initBlur, thresh, edgeThreshold, 0.0f, false,
                                memoryTmp);
                    MatchSiftData(siftData1, siftData2);
                    FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
                    numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0);

                    MatchSiftData(siftData1, siftData3);
                    FindHomography(siftData1, homography, &numMatchesfake, 10000, 0.00f, 0.80f, 5.0);
                    numFitfake = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0);

                    writeToTxt(file, nOctaveLayers, thresh, edgeThreshold, initBlur, siftData1.numPts, siftData2.numPts,
                               numFit, 100.0f * numFit / std::min(siftData1.numPts, siftData2.numPts),
                               siftData1.numPts, siftData3.numPts,
                               numFitfake, 100.0f * numFitfake / std::min(siftData1.numPts, siftData3.numPts),
                               100.0f * numFit / std::min(siftData1.numPts, siftData2.numPts) -
                               100.0f * numFitfake / std::min(siftData1.numPts, siftData3.numPts));
                    file << endl;
                    initBlur -= 0.1f;
                }
                edgeThreshold -= 0.5;
            }
            thresh -= 0.1f;
        }

    }
    file.close();
    FreeSiftTempMemory(memoryTmp);
    // Print out and store summary data
    //PrintMatchData(siftData1, siftData2, oriCuda);
    //cv::imwrite("data/original_pts.pgm", original);

    //MatchAll(siftData1, siftData2, homography);
    // Free Sift data from device
    FreeSiftData(siftData1);
    FreeSiftData(siftData2);
    FreeSiftData(siftData3);
    return 0;
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography) {
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    int numPts1 = siftData1.numPts;
    int numPts2 = siftData2.numPts;
    int numFound = 0;
#if 1
    homography[0] = homography[4] = -1.0f;
    homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
    homography[2] = 1279.0f;
    homography[5] = 959.0f;
#endif
    for (int i = 0; i < numPts1; i++) {
        float *data1 = sift1[i].data;
        std::cout << i << ":" << sift1[i].scale << ":" << (int) sift1[i].orientation << " " << sift1[i].xpos << " "
                  << sift1[i].ypos << std::endl;
        bool found = false;
        for (int j = 0; j < numPts2; j++) {
            float *data2 = sift2[j].data;
            float sum = 0.0f;
            for (int k = 0; k < 128; k++)
                sum += data1[k] * data2[k];
            float den = homography[6] * sift1[i].xpos + homography[7] * sift1[i].ypos + homography[8];
            float dx = (homography[0] * sift1[i].xpos + homography[1] * sift1[i].ypos + homography[2]) / den -
                       sift2[j].xpos;
            float dy = (homography[3] * sift1[i].xpos + homography[4] * sift1[i].ypos + homography[5]) / den -
                       sift2[j].ypos;
            float err = dx * dx + dy * dy;
            if (err < 100.0f) // 100.0
                found = true;
            if (err < 100.0f || j == sift1[i].match) { // 100.0
                if (j == sift1[i].match && err < 100.0f)
                    std::cout << " *";
                else if (j == sift1[i].match)
                    std::cout << " -";
                else if (err < 100.0f)
                    std::cout << " +";
                else
                    std::cout << "  ";
                std::cout << j << ":" << sum << ":" << (int) sqrt(err) << ":" << sift2[j].scale << ":"
                          << (int) sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " "
                          << (int) dx << " " << (int) dy << std::endl;
            }
        }
        std::cout << std::endl;
        if (found)
            numFound++;
    }
    std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
    std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
    std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
    std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img) {
    int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    float *h_img = img.h_data;
    int w = img.width;
    int h = img.height;
    std::cout << std::setprecision(3);
    for (int j = 0; j < numPts; j++) {
        int k = sift1[j].match;
        if (sift1[j].match_error < 5) {
            float dx = sift2[k].xpos - sift1[j].xpos;
            float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
            if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
    std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
    std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
    std::cout << "scale=" << sift1[j].scale << "  ";
    std::cout << "error=" << (int)sift1[j].match_error << "  ";
    std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
    std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
#if 1
            int len = (int) (fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
            for (int l = 0; l < len; l++) {
                int x = (int) (sift1[j].xpos + dx * l / len);
                int y = (int) (sift1[j].ypos + dy * l / len);
                h_img[y * w + x] = 255.0f;
            }
#endif
        }
        int x = (int) (sift1[j].xpos + 0.5);
        int y = (int) (sift1[j].ypos + 0.5);
        int s = std::min(x, std::min(y, std::min(w - x - 2, std::min(h - y - 2, (int) (1.41 * sift1[j].scale)))));
        int p = y * w + x;
        p += (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] = 0.0f;
        p -= (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] = 255.0f;
    }
    std::cout << std::setprecision(6);
}