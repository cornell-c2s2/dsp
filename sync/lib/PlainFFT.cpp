/*
    FFT library
    Copyright (C) 2010 Didier Longueville

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "PlainFFT.h"
#include <math.h>   // ensure M_PI, sqrt, etc. are defined
#include <stdlib.h> // abs, if needed

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define twoPi (2.0 * M_PI)
#define fourPi (4.0 * M_PI)

PlainFFT::PlainFFT(void)
{
    /* Constructor */
}

PlainFFT::~PlainFFT(void)
{
    /* Destructor */
}

uint8_t PlainFFT::Revision(void)
{
    return (FFT_LIB_REV);
}

void PlainFFT::Compute(float *vReal, float *vImag, uint16_t samples, uint8_t dir)
{
    /* Computes in-place complex-to-complex FFT */
    /* Reverse bits */
    uint16_t j = 0;
    for (uint16_t i = 0; i < (samples - 1); i++)
    {
        if (i < j)
        {
            Swap(&vReal[i], &vReal[j]);
            Swap(&vImag[i], &vImag[j]);
        }
        uint16_t k = (samples >> 1);
        while (k <= j)
        {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Compute the FFT  */
    float c1 = -1.0;
    float c2 = 0.0;
    uint16_t l2 = 1;
    uint8_t exponent = Exponent(samples);
    for (uint8_t l = 0; l < exponent; l++)
    {
        uint16_t l1 = l2;
        l2 <<= 1;
        float u1 = 1.0;
        float u2 = 0.0;
        for (uint16_t m = 0; m < l1; m++)
        { // renamed j to m for clarity
            for (uint16_t i = m; i < samples; i += l2)
            {
                uint16_t i1 = i + l1;
                float t1 = u1 * vReal[i1] - u2 * vImag[i1];
                float t2 = u1 * vImag[i1] + u2 * vReal[i1];
                vReal[i1] = vReal[i] - t1;
                vImag[i1] = vImag[i] - t2;
                vReal[i] += t1;
                vImag[i] += t2;
            }
            float z = ((u1 * c1) - (u2 * c2));
            u2 = ((u1 * c2) + (u2 * c1));
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == FFT_FORWARD)
        {
            c2 = -c2;
        }
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for reverse transform */
    if (dir != FFT_FORWARD)
    {
        for (uint16_t i = 0; i < samples; i++)
        {
            vReal[i] /= samples;
            vImag[i] /= samples;
        }
    }
}

void PlainFFT::ComplexToMagnitude(float *vReal, float *vImag, uint16_t samples)
{
    // Use uint16_t for the loop variable to avoid overflow issues
    for (uint16_t i = 0; i < samples; i++)
    {
        float rr = vReal[i] * vReal[i];
        float ii = vImag[i] * vImag[i];
        vReal[i] = sqrt(rr + ii);
    }
}

void PlainFFT::Windowing(float *vData, uint16_t samples, uint8_t windowType, uint8_t dir)
{
    float samplesMinusOne = (float(samples) - 1.0);
    for (uint16_t i = 0; i < (samples >> 1); i++)
    {
        float indexMinusOne = (float)i;
        float ratio = indexMinusOne / samplesMinusOne;
        float weighingFactor = 1.0;
        /* Compute and record weighting factor */
        switch (windowType)
        {
        case FFT_WIN_TYP_RECTANGLE: /* rectangle (box car) */
            weighingFactor = 1.0;
            break;
        case FFT_WIN_TYP_HAMMING: /* hamming */
            weighingFactor = 0.54 - (0.46 * cos(twoPi * ratio));
            break;
        case FFT_WIN_TYP_HANN: /* hann */
            // Correct Hann window formula: 0.5 * (1 - cos(2πn/(N-1)))
            weighingFactor = 0.5 * (1.0 - cos(twoPi * ratio));
            break;
        case FFT_WIN_TYP_TRIANGLE: /* triangle (Bartlett) */
            weighingFactor = 1.0 - ((2.0 * fabs(indexMinusOne - (samplesMinusOne / 2.0))) / samplesMinusOne);
            break;
        case FFT_WIN_TYP_BLACKMAN: /* blackmann */
            weighingFactor = 0.42323 - (0.49755 * cos(twoPi * ratio)) + (0.07922 * cos(fourPi * ratio));
            break;
        case FFT_WIN_TYP_FLT_TOP: /* flat top */
            weighingFactor = 0.2810639 - (0.5208972 * cos(twoPi * ratio)) + (0.1980399 * cos(fourPi * ratio));
            break;
        case FFT_WIN_TYP_WELCH: /* welch */
            weighingFactor = 1.0 - (((indexMinusOne - (samplesMinusOne / 2.0)) * (indexMinusOne - (samplesMinusOne / 2.0))) / ((samplesMinusOne / 2.0) * (samplesMinusOne / 2.0)));
            break;
        default:
            weighingFactor = 1.0; // default no window
            break;
        }

        if (dir == FFT_FORWARD)
        {
            vData[i] *= weighingFactor;
            vData[samples - (i + 1)] *= weighingFactor;
        }
        else
        {
            vData[i] /= (weighingFactor == 0.0 ? 1.0 : weighingFactor);
            vData[samples - (i + 1)] /= (weighingFactor == 0.0 ? 1.0 : weighingFactor);
        }
    }
}

float PlainFFT::MajorPeak(float *vD, uint16_t samples, float samplingFrequency)
{
    float maxY = 0;
    uint16_t IndexOfMaxY = 0;
    for (uint16_t i = 1; i < ((samples >> 1) - 1); i++)
    {
        if ((vD[i - 1] < vD[i]) && (vD[i] > vD[i + 1]))
        {
            if (vD[i] > maxY)
            {
                maxY = vD[i];
                IndexOfMaxY = i;
            }
        }
    }
    float delta = 0.5 * ((vD[IndexOfMaxY - 1] - vD[IndexOfMaxY + 1]) / (vD[IndexOfMaxY - 1] - (2.0 * vD[IndexOfMaxY]) + vD[IndexOfMaxY + 1]));
    float interpolatedX = ((IndexOfMaxY + delta) * samplingFrequency) / (samples - 1);
    /* returned value: interpolated frequency peak apex */
    return (interpolatedX);
}

/* Private functions */

void PlainFFT::Swap(float *x, float *y)
{
    float temp = *x;
    *x = *y;
    *y = temp;
}

uint8_t PlainFFT::Exponent(uint16_t value)
{
    /* Computes the Exponent of a powered 2 value */
    uint8_t result = 0;
    while (((value >> result) & 1) != 1)
        result++;
    return (result);
}
