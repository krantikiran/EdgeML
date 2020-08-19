// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>
#include <limits.h>
#include <fstream>
#include "datatypes.h"
#include "library_fixed.h"
#include "profile.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAddNN(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC)
{

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++)
	{
		// MYINT b = getIntFeature(k);
		MYINT b = B[k * 1][0];
#ifdef FASTAPPROX
		b = b / shrB;
#endif

		MYITE idx = Aidx[ite_idx];
		while (idx != 0)
		{
			MYINT a = Aval[ite_val];
#ifdef FASTAPPROX
			a = a / shrA;

			MYINT c = a * b;
			c = c / shrC;
#else
			MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
#endif

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			a = a / shrA;
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrB * (int64_t)shrA));
#endif
		}
	}
	return;
}



int32_t fixedTanH(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter){
	int32_t fixedExp(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter);
	int32_t scale_factor = scale_out;
    int32_t exp_res = 0;
    int32_t ans = 0;
	int64_t ans64 = 0;
    if (x < 0)
    {
		 if(x < (-10*scale_in))
            exp_res = 0;
        else
            exp_res = fixedExp(2*x, scale_in, scale_out, iter);

		// ans64 = int64_t((int64_t((int64_t)exp_res - (int64_t)scale_factor))*int64_t(scale_out));
		// ans = ans64/(int64_t((int64_t)scale_factor + (int64_t)exp_res));
		ans = division(exp_res - scale_factor, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
    }
    else
    {
		 if ((-1*x) < (-10*scale_in))
            exp_res = 0;
        else
            exp_res = fixedExp(-2*x, scale_in, scale_out, iter);

        	// ans64 = int64_t((int64_t((int64_t)scale_factor-(int64_t)exp_res))*int64_t(scale_out));
			// ans = ans64/(int64_t((int64_t)scale_factor + (int64_t)exp_res));
			ans = division(scale_factor - exp_res, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
    }
    return ans;
}


// A = tanh(A)
void TanH(MYINT *A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT *B)
{
	int32_t iter = 2;
	// std::ifstream iterfile("/home/krantikiran/msr/EdgeML/Tools/SeeDot/iterinfile");
	// iterfile >> iter;
	// iterfile.close();

	int32_t SCALE_IN = log2(scale_in);
	int32_t SCALE_OUT = log2(scale_out);

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
#ifdef FLOATEXP
			// float x = float(A[i * J + j]) / scale_in;
			// B[i * J + j] = tanh(x)*scale_out;
			// updateRangeOfTanH(x, SCALE_IN, SCALE_OUT);
			
			int32_t tanh_res;
			tanh_res = fixedTanH(A[i * J + j], scale_in, scale_out, iter);
			B[i * J + j] = tanh_res;
#else
			MYINT x = A[i * J + j], y;

			if (x >= scale_in)
				y = scale_in;
			else if (x <= -scale_in)
				y = -scale_in;
			else
				y = x;

			MYINT scale_diff = scale_out / scale_in;

			y *= scale_diff;

			B[i * J + j] = y;
#endif
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index)
{

	MYINT max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT x = A[i * J + j];

			if (max < x)
			{
				maxIndex = counter;
				max = x;
			}

			counter++;
		}
	}

	*index = maxIndex;

	return;
}

// B = reverse(A, axis)
void Reverse2(MYINT *A, MYINT axis, MYINT I, MYINT J, MYINT *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{	
			MYINT i_prime = (axis == 0 ? (I-1-i) : i);
			MYINT j_prime = (axis == 1 ? (J-1-j) : j); 

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// A = A^T
void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{

	MYINT a = *A;
#ifdef FASTAPPROX
	a = a / shrA;
#endif

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrA * (int64_t)shrB));
#endif
		}
	}

	return;
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE co = 0; co < CO; co++)
				{

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++)
					{
						for (MYITE wf = 0; wf < WF; wf++)
						{
							for (MYITE ci = 0; ci < CI; ci++)
							{
								MYINT a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								MYINT b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

							#ifdef FASTAPPROX
								a = a / shrA;
								b = b / shrB;

								tmp[counter] = a * b;
							#else
								int64_t temp = (((int64_t) a) * ((int64_t)b)) / (((int64_t)shrA) * ((int64_t)shrB));
								tmp[counter] = Saturate<MYINT>(temp);
							#endif
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2))
					{
						if (depth >= H1)
							shr = false;

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++)
						{
							MYINT sum;
							if (p < (count >> 1))
							{
								if (shr)
									sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
								else
									sum = tmp[2 * p] + tmp[(2 * p) + 1];
							}
							else if ((p == (count >> 1)) && ((count & 1) == 1))
							{
								if (shr)
									sum = tmp[2 * p] / 2;
								else
									sum = tmp[2 * p];
							}
							else
								sum = 0;

							tmp[p] = sum;
						}
						count = (count + 1) >> 1;

						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	MYITE HOffsetL = HDL*(HF/2) - HPADL;
	MYITE WOffsetL = WDL*(WF/2) - WPADL;
	MYITE HOffsetR = HDL*(HF/2) - HPADR;
	MYITE WOffsetR = WDL*(WF/2) - WPADR;

	for(MYITE n = 0; n < N; n++) {
		for(MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for(MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for(MYITE g = 0; g < G; g++) {
					for(MYITE co = 0; co < COUTF; co ++) {

						MYITE counter = 0;
						for(MYITE hf = -(HF/2); hf <= HF/2; hf++) {
							for(MYITE wf = -(WF/2); wf <= WF/2; wf++) {
								for(MYITE ci = 0; ci < CINF; ci++) {

									MYINT a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									MYINT b = B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];

								#ifdef FASTAPPROX
									a = a / shrA;
									b = b / shrB;

									tmp[counter] = a * b;
								#else
									int64_t temp = (((int64_t) a) * ((int64_t)b)) / (((int64_t)shrA) * ((int64_t)shrB));
									tmp[counter] = Saturate<MYINT>(temp);
								#endif

									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1)
								shr = false;

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								MYINT sum;
								if (p < (count >> 1)) {
									if (shr)
										sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
									else
										sum = tmp[2 * p] + tmp[(2 * p) + 1];
								}
								else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									if (shr)
										sum = tmp[2 * p] / 2;
									else
										sum = tmp[2 * p];
								}
								else
									sum = 0;

								tmp[p] = sum;
							}
							count = (count + 1) >> 1;

							depth++;
						}

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
					}
				}
			}
		}
	}
}


// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					MYINT b = B[c];
					b = b / shrB;

					MYINT res;
					if (add)
						res = Saturate<MYINT>(a / shrC + b / shrC);
					else
						res = Saturate<MYINT>(a / shrC - b / shrC);

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			MYINT a = A[h * W + w];
			a = a / shrA;

			MYINT b = B[w];
			b = b / shrB;

			MYINT res;
			if (add)
				res = Saturate<MYINT>(a / shrC + b / shrC);
			else
				res = Saturate<MYINT>(a / shrC - b / shrC);

			X[h * W + w] = res;
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(MYINT *A, MYINT H, MYINT W)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			MYINT a = A[h * W + w];
			if (a < 0)
				a = 0;

			A[h * W + w] = a;
		}
	}

	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR)
{
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE ho = 0; ho < HO; ho++)
		{
			for (MYITE wo = 0; wo < WO; wo++)
			{
				for (MYITE c = 0; c < C; c++)
				{

					MYINT max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++)
					{
						for (MYITE ws = 0; ws < FW; ws++)
						{
							MYINT a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}

	return;
}

// A = Normalise(A)
void NormaliseL2(MYINT* A, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
		 
				// calculate the sum square
				int32_t sumSquare = 0;
				MYINT shrAdiv = (1<<shrA);

				for (MYITE c = 0; c < C; c++) {
#ifdef FASTAPPROX
				MYINT tmp = (A[n * H * W * C + h * W * C + w * C + c] / shrAdiv);
				sumSquare += tmp*tmp;
#else           
				int32_t tmp = A[n * H * W * C + h * W * C + w * C + c];
				sumSquare += (((tmp*tmp)/shrAdiv)/shrAdiv);						
#endif
				}
				

				// calculate the inverse square root of sumSquare
				MYINT yLow = 1;

				// yHigh: A number of length shrA with all 1s in binary representation e.g. for shrA=8 --> y_high = 0b11111111
				MYINT yHigh = (1<<shrA - 1);   

				// one: value of 1 with same scale as y*y*sumSquare
				// scale of sumSquare = 2*scale_in + 2*shrA
				// since we assume scale of y = 1 - shrA
				// scale of y*y*sumSquare =  2*scale_in + 2*shrA + 2(1-shrA) = 2*scale_in + 2
				int32_t one = ( 1<< (-(2*scaleA + 2)) ); 

				//binary search for the inverse square root 
				while( yLow+1 < yHigh){

					//using int32_t sotherwise (y*y*sumSquare) will overflow
					MYINT yMid = ((yHigh + yLow)>>1);

					int64_t cmpValue = (int64_t)sumSquare*yMid*yMid;

					if(cmpValue > one){
						yHigh = yMid;	
					}	
					else {
						yLow = yMid;
					}
				}
				MYINT inverseNorm = yLow;


				// multiply all elements by the 1/sqrt(sumSquare)
				for (MYITE c = 0; c < C; c++){
						A[n * H * W * C + h * W * C + w * C + c]  = (A[n * H * W * C + h * W * C + w * C + c]  / shrAdiv)*inverseNorm;  
				}	
			}					
		}
	}
	return;
}

#define __LOG2E__ (1.44269504089 * (scale_in))
#define __LOGE2__ (0.69314718056 * (scale_in))

uint32_t computeULPErr(float calc, float actual)
{
    int32_t calc_xx =  *((int32_t*)&calc);
    calc_xx = calc_xx<0?INT_MIN - calc_xx: calc_xx;

    int32_t act_yy = *((int32_t*)&actual);
    act_yy = act_yy<0?INT_MIN - act_yy: act_yy;
  
    uint32_t ulp_err = (calc_xx-act_yy)>0?(calc_xx-act_yy):(act_yy-calc_xx);

    return ulp_err;
}

int32_t fixed_point_round(int32_t x, int32_t scale){
    int32_t mask = (scale>>1);
    
    int32_t val = (mask & x);
    if ((mask & x) == 0){
        x = x/scale;
    }
    else
    {   
        x = (x/scale) + 1;
    }
    
    return x;
}

int32_t fixedExp(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter){
	// static int global_iter = -1;
	// int64_t t1, t2;
	int32_t scale_factor = scale_in;
    if (x < -10*scale_factor)
        return 0;
    // t1 = int64_t(x) * int64_t(__LOG2E__);
    // int32_t N = fixed_point_round(t1/scale_in , scale_in);

    // t2 = int64_t((scale_factor*N)) * int64_t(__LOGE2__);
    // int32_t d = x -  (t2/scale_in) ;
    
    // int count = 1;
    // int32_t y = scale_factor; 
    // int32_t ans = scale_factor;
	// if (global_iter == -1){
	// 	std::ifstream iterfile("/home/krantikiran/msr/EdgeML/Tools/SeeDot/iterinfile");
	// 	iterfile >> global_iter;
	// 	iterfile.close();
	// }
	// for(MYITE i = 0;i<global_iter;i++){
	// 	int64_t big_y = y;
	// 	big_y *=d;
	// 	y = big_y/scale_in;
	// 	y = y/count;
	// 	ans += y;
	// 	count++;
	// }

	// int32_t scale_diff = scale_out/scale_in;
	// // N += scale_diff;

    // if (N>0)
    //     ans <<= N;
    // else if (N < 0)
    //     ans >>= (-1*N);
    
    // // cout << " ans "<<ans<<endl;

    // return ans * scale_diff;
	int32_t expTableTmp(int32_t a, int32_t scale_in, int32_t scale_out);
	return expTableTmp(x, log2(scale_in), log2(scale_out));
}

int32_t expTableTmp(int32_t a, int32_t scale_in, int32_t scale_out){
    // const int32_t expTable32A[256] = {1073741824, 947573824, 836230976, 737971264, 651257344, 574732608, 507199712, 447602176, 395007552, 348592928, 307632192, 271484448, 239584192, 211432304, 186588352, 164663648, 145315152, 128240176, 113171552, 99873544, 88138096, 77781600, 68642016, 60576368, 53458456, 47176924, 41633488, 36741424, 32424194, 28614250, 25251988, 22284800, 19666268, 17355420, 15316105, 13516415, 11928194, 10526594, 9289687, 8198120, 7234815, 6384702, 5634480, 4972411, 4388137, 3872517, 3417484, 3015919, 2661540, 2348800, 2072809, 1829247, 1614305, 1424619, 1257222, 1109494, 979125, 864075, 762543, 672942, 593869, 524088, 462506, 408160, 360200, 317875, 280524, 247561, 218472, 192801, 170146, 150153, 132510, 116939, 103199, 91072, 80371, 70927, 62593, 55238, 48747, 43019, 37964, 33503, 29567, 26092, 23026, 20321, 17933, 15826, 13966, 12325, 10877, 9599, 8471, 7475, 6597, 5822, 5137, 4534, 4001, 3531, 3116, 2750, 2427, 2141, 1890, 1668, 1472, 1299, 1146, 1011, 892, 787, 695, 613, 541, 477, 421, 372, 328, 289, 255, 225, 199, 175, 155, 136, 120, 106, 94, 83, 73, 64, 57, 50, 44, 39, 34, 30, 26, 23, 20, 18, 16, 14, 12, 11, 9, 8, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // const int32_t expTable32B[256] = {1073741824, 1073217664, 1072693760, 1072170112, 1071646720, 1071123584, 1070600704, 1070078080, 1069555712, 1069033600, 1068511744, 1067990144, 1067468736, 1066947648, 1066426816, 1065906240, 1065385920, 1064865792, 1064345984, 1063826432, 1063307072, 1062788032, 1062269248, 1061750656, 1061232384, 1060714304, 1060196480, 1059678976, 1059161664, 1058644608, 1058127808, 1057611264, 1057094976, 1056578944, 1056063168, 1055547648, 1055032384, 1054517376, 1054002560, 1053488064, 1052973760, 1052459776, 1051945984, 1051432448, 1050919168, 1050406144, 1049893376, 1049380864, 1048868608, 1048356608, 1047844864, 1047333312, 1046822080, 1046311040, 1045800256, 1045289728, 1044779456, 1044269440, 1043759680, 1043250176, 1042740864, 1042231872, 1041723072, 1041214528, 1040706240, 1040198208, 1039690432, 1039182912, 1038675584, 1038168576, 1037661760, 1037155200, 1036648960, 1036142848, 1035637056, 1035131520, 1034626176, 1034121152, 1033616320, 1033111744, 1032607424, 1032103360, 1031599488, 1031095936, 1030592576, 1030089472, 1029586624, 1029084032, 1028581696, 1028079552, 1027577664, 1027076032, 1026574656, 1026073536, 1025572672, 1025072000, 1024571584, 1024071424, 1023571520, 1023071872, 1022572416, 1022073280, 1021574336, 1021075648, 1020577152, 1020078976, 1019580992, 1019083264, 1018585792, 1018088576, 1017591552, 1017094848, 1016598336, 1016102080, 1015606016, 1015110272, 1014614720, 1014119424, 1013624384, 1013129536, 1012635008, 1012140672, 1011646592, 1011152704, 1010659136, 1010165760, 1009672640, 1009179712, 1008687104, 1008194688, 1007702528, 1007210624, 1006718912, 1006227456, 1005736256, 1005245312, 1004754624, 1004264128, 1003773888, 1003283840, 1002794112, 1002304576, 1001815296, 1001326208, 1000837440, 1000348864, 999860544, 999372416, 998884608, 998396928, 997909568, 997422464, 996935552, 996448896, 995962432, 995476224, 994990272, 994504576, 994019072, 993533888, 993048832, 992564096, 992079552, 991595264, 991111168, 990627392, 990143808, 989660416, 989177344, 988694464, 988211776, 987729408, 987247232, 986765312, 986283584, 985802112, 985320896, 984839872, 984359104, 983878592, 983398336, 982918272, 982438400, 981958848, 981479488, 981000384, 980521472, 980042816, 979564416, 979086208, 978608256, 978130560, 977653056, 977175808, 976698752, 976222016, 975745472, 975269120, 974793024, 974317184, 973841536, 973366144, 972891008, 972416064, 971941376, 971466880, 970992704, 970518656, 970044928, 969571392, 969098048, 968624960, 968152128, 967679488, 967207104, 966734976, 966263040, 965791360, 965319872, 964848640, 964377664, 963906880, 963436352, 962966016, 962495936, 962026112, 961556480, 961087104, 960617920, 960148992, 959680256, 959211776, 958743552, 958275520, 957807744, 957340160, 956872832, 956405696, 955938816, 955472192, 955005760, 954539520, 954073600, 953607808, 953142336, 952677056, 952211968, 951747136, 951282560, 950818176, 950353984, 949890048, 949426368, 948962880, 948499648, 948036608};
    // const int32_t expTable32C[256] = {1073741824, 1073739776, 1073737728, 1073735680, 1073733632, 1073731584, 1073729536, 1073727488, 1073725440, 1073723392, 1073721344, 1073719296, 1073717248, 1073715200, 1073713152, 1073711104, 1073709056, 1073707008, 1073704960, 1073702912, 1073700864, 1073698816, 1073696768, 1073694720, 1073692672, 1073690624, 1073688576, 1073686528, 1073684480, 1073682432, 1073680384, 1073678336, 1073676288, 1073674240, 1073672192, 1073670144, 1073668096, 1073666048, 1073664000, 1073661952, 1073659904, 1073657856, 1073655808, 1073653760, 1073651712, 1073649664, 1073647616, 1073645568, 1073643520, 1073641472, 1073639424, 1073637376, 1073635328, 1073633280, 1073631232, 1073629184, 1073627136, 1073625088, 1073623040, 1073620992, 1073618944, 1073616896, 1073614848, 1073612800, 1073610752, 1073608704, 1073606656, 1073604608, 1073602560, 1073600512, 1073598464, 1073596416, 1073594368, 1073592320, 1073590272, 1073588224, 1073586176, 1073584128, 1073582080, 1073580032, 1073577984, 1073575936, 1073573888, 1073571840, 1073569792, 1073567744, 1073565696, 1073563648, 1073561600, 1073559552, 1073557504, 1073555456, 1073553408, 1073551360, 1073549312, 1073547264, 1073545216, 1073543168, 1073541120, 1073539072, 1073537024, 1073534976, 1073532928, 1073530880, 1073528832, 1073526784, 1073524736, 1073522688, 1073520640, 1073518592, 1073516544, 1073514496, 1073512448, 1073510400, 1073508352, 1073506304, 1073504256, 1073502208, 1073500160, 1073498112, 1073496064, 1073494016, 1073491968, 1073489920, 1073487872, 1073485824, 1073483776, 1073481728, 1073479680, 1073477696, 1073475648, 1073473600, 1073471552, 1073469504, 1073467456, 1073465408, 1073463360, 1073461312, 1073459264, 1073457216, 1073455168, 1073453120, 1073451072, 1073449024, 1073446976, 1073444928, 1073442880, 1073440832, 1073438784, 1073436736, 1073434688, 1073432640, 1073430592, 1073428544, 1073426496, 1073424448, 1073422400, 1073420352, 1073418304, 1073416256, 1073414208, 1073412160, 1073410112, 1073408064, 1073406016, 1073403968, 1073401920, 1073399872, 1073397824, 1073395776, 1073393728, 1073391680, 1073389632, 1073387584, 1073385536, 1073383488, 1073381440, 1073379392, 1073377344, 1073375296, 1073373248, 1073371200, 1073369152, 1073367104, 1073365056, 1073363008, 1073360960, 1073358912, 1073356864, 1073354816, 1073352768, 1073350720, 1073348672, 1073346624, 1073344576, 1073342528, 1073340480, 1073338432, 1073336384, 1073334336, 1073332288, 1073330240, 1073328192, 1073326144, 1073324096, 1073322048, 1073320000, 1073317952, 1073315904, 1073313856, 1073311808, 1073309760, 1073307712, 1073305664, 1073303616, 1073301568, 1073299520, 1073297472, 1073295424, 1073293376, 1073291328, 1073289280, 1073287296, 1073285248, 1073283200, 1073281152, 1073279104, 1073277056, 1073275008, 1073272960, 1073270912, 1073268864, 1073266816, 1073264768, 1073262720, 1073260672, 1073258624, 1073256576, 1073254528, 1073252480, 1073250432, 1073248384, 1073246336, 1073244288, 1073242240, 1073240192, 1073238144, 1073236096, 1073234048, 1073232000, 1073229952, 1073227904, 1073225856, 1073223808, 1073221760, 1073219712};
    // const int32_t expTable32D[128] = {1073741824, 1073741824, 1073741824, 1073741760, 1073741760, 1073741760, 1073741760, 1073741696, 1073741696, 1073741696, 1073741696, 1073741632, 1073741632, 1073741632, 1073741632, 1073741568, 1073741568, 1073741568, 1073741568, 1073741504, 1073741504, 1073741504, 1073741504, 1073741440, 1073741440, 1073741440, 1073741440, 1073741376, 1073741376, 1073741376, 1073741376, 1073741312, 1073741312, 1073741312, 1073741312, 1073741248, 1073741248, 1073741248, 1073741248, 1073741184, 1073741184, 1073741184, 1073741184, 1073741120, 1073741120, 1073741120, 1073741120, 1073741056, 1073741056, 1073741056, 1073741056, 1073740992, 1073740992, 1073740992, 1073740992, 1073740928, 1073740928, 1073740928, 1073740928, 1073740864, 1073740864, 1073740864, 1073740864, 1073740800, 1073740800, 1073740800, 1073740800, 1073740736, 1073740736, 1073740736, 1073740736, 1073740672, 1073740672, 1073740672, 1073740672, 1073740608, 1073740608, 1073740608, 1073740608, 1073740544, 1073740544, 1073740544, 1073740544, 1073740480, 1073740480, 1073740480, 1073740480, 1073740416, 1073740416, 1073740416, 1073740416, 1073740352, 1073740352, 1073740352, 1073740352, 1073740288, 1073740288, 1073740288, 1073740288, 1073740224, 1073740224, 1073740224, 1073740224, 1073740160, 1073740160, 1073740160, 1073740160, 1073740096, 1073740096, 1073740096, 1073740096, 1073740032, 1073740032, 1073740032, 1073740032, 1073739968, 1073739968, 1073739968, 1073739968, 1073739904, 1073739904, 1073739904, 1073739904, 1073739840, 1073739840, 1073739840, 1073739840, 1073739776};
    // int val1 = a%128;
    // a = a/128;
    // int val2 = a%256;
    // a = a/256;
    // int val3 = a%256;
    // a = a/256;
    // int val4 = a;
    // int64_t temp1 = int64_t(expTable32A[val4])*int64_t(expTable32B[val3]);
    // temp1 = temp1/1073741824;
    // int64_t temp2 = int64_t(expTable32C[val2])*int64_t(expTable32D[val1]);
    // temp2 = temp2/1073741824;
    // temp1 = temp1 * temp2;
    // int32_t expval = temp1/1073741824;
    // return expval;
	a = (a == INT_MIN)? INT_MAX: -a;
    int32_t exp_valD, exp_valC, exp_valB, exp_valA;
	int32_t val1 = a%128;
    a = a/128;
    int32_t val2 = a%256;
    a = a/256;
    int32_t val3 = a%256;
    a = a/256;
    int32_t val4 = a;
	
	float factor2, factor3, factor4;
    factor2 = (scale_in-7)>0?float(1<<(scale_in-7)):1.0/float(1<<(7-scale_in));
    factor3 = (scale_in-15)>0?float(1<<(scale_in-15)):1.0/float(1<<(15-scale_in));
    factor4 = (scale_in-23)>0?float(1<<(scale_in-23)):1.0/float(1<<(23-scale_in));

    exp_valD = int32_t(exp(float(-val1)/(1<<scale_in))*(1<<scale_out));
	exp_valC = int32_t(exp(float(-val2)/factor2)*(1<<scale_out));
	exp_valB = int32_t(exp(float(-val3)/factor3)*(1<<scale_out));
	exp_valA = int32_t(exp(float(-val4)/factor4)*(1<<scale_out));
    
	int64_t temp1 = int64_t(exp_valA)*int64_t(exp_valB);
    temp1 = temp1>>scale_out;
    int64_t temp2 = int64_t(exp_valC)*int64_t(exp_valD);
    temp2 = temp2>>scale_out;
    temp1 = temp1 * temp2;
    int32_t expval = temp1>>scale_out;
    return expval;
}

// B = exp(A)
void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B)
{
	int32_t iter= 2;
	// std::ifstream iterfile("/home/krantikiran/msr/EdgeML/Tools/SeeDot/iterinfile");
	// iterfile >> iter;
	// iterfile.close();
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			//B[i * J + j] = (MYINT)(exp(float(A[i * J + j])/float(shrA))* float(shrB));
			// updateRangeOfExpFixed(float(float(A[i * J + j])/float(shrA)), SCALE_IN, SCALE_OUT);
			int32_t exp_res = fixedExp(A[i * J + j], shrA, shrB, iter);
			B[i * J + j] = ((MYINT)exp_res);
		}
	}

	return;
}

int32_t fixedSigmoid(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter){

	int32_t scale_factor = scale_out;
    int32_t exp_res = 0;
    int32_t ans = 0;
	int64_t ans64 = 0;
    if (x < 0)
    {
        exp_res = fixedExp(x, scale_in, scale_out, iter);

		// ans64 = (int64_t(exp_res) * int64_t(scale_out));
		// ans = ans64/((int64_t)scale_factor + (int64_t)exp_res);
		ans = division(exp_res, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
    }
    else
    {
        exp_res = fixedExp(-x, scale_in, scale_out, iter);
		// ans64 = (int64_t(scale_factor) * (int64_t)scale_out);
		// ans = ans64/((int64_t)scale_factor + (int64_t)exp_res);
		ans = division(scale_factor, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
	}
    return ans;
}


// B = Sigmoid(A)
void Sigmoid(MYINT *A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT *B)
{
	int32_t iter = 2;
	// int32_t SCALE_IN = log2(scale_in);
	// int32_t SCALE_OUT = log2(scale_out);
	// if(SCALE_IN == 31 || SCALE_OUT == 32)
	// 	std::cout<<"shit "<<std::endl;
	MYINT scale_diff = scale_out / scale_in;

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
#ifdef FLOATEXP
			// float x = float(A[i * J + j]) / scale_in;
			// float y = 1 / (1 + exp(-x));
			// MYINT z = MYINT(y * scale_out);
			// B[i * J + j] = z;
			
			int32_t 
			sigmoid_res = fixedSigmoid(A[i * J + j], scale_in, scale_out, iter); 
			B[i * J + j] = sigmoid_res;
			
#else
			MYINT x = A[i * J + j];

			x = (x / div) + add;

			MYINT y;
			if (x >= sigmoid_limit)
				y = sigmoid_limit;
			else if (x <= 0)
				y = 0;
			else
				y = x;

			y = y * scale_diff;

			B[i * J + j] = y;
#endif
		}
	}

	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for(MYITE k = 0; k < K; k++) 
			{
				for(MYITE l = 0; l < L; l++) 
				{
					MYINT a = A[i * J * K * L + j * K * L + k * L + l];
					A[i * J * K * L + j * K * L + k * L + l] = a / scale;
				}
			}
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT scale)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT scale)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}

	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for(MYITE k = 0; k < K; k++) 
			{
				for(MYITE l = 0; l < L; l++) 
				{
					MYINT a = A[i * J * K * L + j * K * L + k * L + l];
					A[i * J * K * L + j * K * L + k * L + l] = a * scale;
				}
			}
		}
	}

	return;
}

MYINT treeSum(MYINT *arr, MYINT count, MYINT height_shr, MYINT height_noshr)
{
	if (count == 1)
		return arr[0];

	bool shr = true;

	for (MYITE depth = 0; depth < (height_shr + height_noshr); depth++)
	{
		if (depth >= height_shr)
			shr = false;

		for (MYITE index = 0; index < (count / 2); index++)
		{
			MYINT sum = arr[2 * index] + arr[(2 * index) + 1];

			if (shr)
				arr[index] = sum / 2;
			else
				arr[index] = sum;
		}

		if (count % 2 == 1)
		{
			MYITE index = (count / 2) + 1;
			if (shr)
				arr[index - 1] = arr[count - 1] / 2;
			else
				arr[index - 1] = arr[count - 1];
		}

		// Debugging
		if (count % 2 == 1)
			arr[count / 2 + 1] = 0;
		else
			arr[count / 2] = 0;

		count = (count + 1) >> 1;
	}

	return arr[0];
}
