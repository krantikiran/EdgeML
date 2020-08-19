#ifndef __FIXED__MATH__UTILS__
#define __FIXED__MATH__UTILS__

#include "library_fixed_div.h"
#include <stdint.h>
#include "datatypes.h"


#define __LOG2E__ (1.44269504089 * (scale_in))
#define __LOGE2__ (0.69314718056 * (scale_in))

#define TABLE_EXP
#define TABLE_DIV

int32_t fixedExp(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter);
int32_t fixedTanH(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter);
uint32_t computeULPErr(float calc, float actual);
int32_t fixed_point_round(int32_t x, int32_t scale);
int32_t expTableTmp(int32_t a, int32_t scale_in, int32_t scale_out);
int32_t fixedSigmoid(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter);


#endif