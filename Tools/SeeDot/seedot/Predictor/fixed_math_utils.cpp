#include "fixed_math_utils.h"

int32_t fixedSigmoid(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter){

	int32_t scale_factor = scale_out;
    int32_t exp_res = 0;
    int32_t ans = 0;
	int64_t ans64 = 0;
    if (x < 0)
    {
        exp_res = fixedExp(x, scale_in, scale_out, iter);
#ifndef TABLE_DIV
		ans64 = (int64_t(exp_res) * int64_t(scale_out));
		ans = ans64/((int64_t)scale_factor + (int64_t)exp_res);
#else
		ans = division(exp_res, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
#endif
    }
    else
    {
        exp_res = fixedExp(-x, scale_in, scale_out, iter);
#ifndef TABLE_DIV
		ans64 = (int64_t(scale_factor) * (int64_t)scale_out);
		ans = ans64/((int64_t)scale_factor + (int64_t)exp_res);
#else		
        ans = division(scale_factor, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
#endif
	}
    return ans;
}

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

	int32_t scale_factor = scale_in;
    if (x < -10*scale_factor)
        return 0;

	/*************Taylor series based exp***************************/
#ifndef TABLE_EXP 
    static int global_iter = -1;
	int64_t t1, t2;
    t1 = int64_t(x) * int64_t(__LOG2E__);
    int32_t N = fixed_point_round(t1/scale_in , scale_in);

    t2 = int64_t((scale_factor*N)) * int64_t(__LOGE2__);
    int32_t d = x -  (t2/scale_in) ;
    
    int count = 1;
    int32_t y = scale_factor; 
    int32_t ans = scale_factor;
	if (global_iter == -1){
		std::ifstream iterfile("/home/krantikiran/msr/EdgeML/Tools/SeeDot/iterinfile");
		iterfile >> global_iter;
		iterfile.close();
	}
	for(MYITE i = 0;i<global_iter;i++){
		int64_t big_y = y;
		big_y *=d;
		y = big_y/scale_in;
		y = y/count;
		ans += y;
		count++;
	}

	int32_t scale_diff = scale_out/scale_in;
	// N += scale_diff;

    if (N>0)
        ans <<= N;
    else if (N < 0)
        ans >>= (-1*N);
    
    // cout << " ans "<<ans<<endl;

    return ans * scale_diff;
    
#else
    /*******************Table Based Exp********************/
	return expTableTmp(x, log2(scale_in), log2(scale_out));
#endif
}

int32_t expTableTmp(int32_t a, int32_t scale_in, int32_t scale_out){
	a = (a == INT_MIN)? INT_MAX: -a;
    int32_t exp_valD, exp_valC, exp_valB, exp_valA;
	int32_t val1 = a%128;
    a = a/128;
    int32_t val2 = a%256;
    a = a/256;
    int32_t val3 = a%256;
    a = a/256;
    int32_t val4 = a;
	/**************The table part emulation ******************/
	float factor2, factor3, factor4;
    factor2 = (scale_in-7)>0?float(1<<(scale_in-7)):1.0/float(1<<(7-scale_in));
    factor3 = (scale_in-15)>0?float(1<<(scale_in-15)):1.0/float(1<<(15-scale_in));
    factor4 = (scale_in-23)>0?float(1<<(scale_in-23)):1.0/float(1<<(23-scale_in));

    exp_valD = int32_t(exp(float(-val1)/(1<<scale_in))*(1<<scale_out));
	exp_valC = int32_t(exp(float(-val2)/factor2)*(1<<scale_out));
	exp_valB = int32_t(exp(float(-val3)/factor3)*(1<<scale_out));
	exp_valA = int32_t(exp(float(-val4)/factor4)*(1<<scale_out));
    /**************The table part emulation ended******************/
    
	int64_t temp1 = int64_t(exp_valA)*int64_t(exp_valB);
    temp1 = temp1>>scale_out;
    int64_t temp2 = int64_t(exp_valC)*int64_t(exp_valD);
    temp2 = temp2>>scale_out;
    temp1 = temp1 * temp2;
    int32_t expval = temp1>>scale_out;
    return expval;
}


int32_t fixedTanH(int32_t x, int32_t scale_in, int32_t scale_out, int32_t iter){
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

#ifndef TABLE_DIV
		ans64 = int64_t((int64_t((int64_t)exp_res - (int64_t)scale_factor))*int64_t(scale_out));
		ans = ans64/(int64_t((int64_t)scale_factor + (int64_t)exp_res));
#else
		ans = division(exp_res - scale_factor, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
#endif
    }
    else
    {
		 if ((-1*x) < (-10*scale_in))
            exp_res = 0;
        else
            exp_res = fixedExp(-2*x, scale_in, scale_out, iter);
#ifndef TABLE_DIV
        	ans64 = int64_t((int64_t((int64_t)scale_factor-(int64_t)exp_res))*int64_t(scale_out));
			ans = ans64/(int64_t((int64_t)scale_factor + (int64_t)exp_res));
#else
			ans = division(scale_factor - exp_res, scale_factor + exp_res, log2(scale_out), 0b11111111, 8);
#endif    
    }
    return ans;
}
