#include "cnn.h"
#include <stdbool.h>
void input_load_conv(float* input, float input_buffer[Tn][Tr + 4][Tc + 4],
	int TR, int TC, int TN, int offset, bool enable)
{
	//	float input_tmp_buffer[n*r*c];

	if (enable)
		return;

	float* input_tmp_buffer = (float*)calloc(TN * TR * TC, sizeof(float));

	memcpy((float*)input_tmp_buffer, (float*)(input + offset), TN * TR * TC * sizeof(float));
	//printf("%d\n", TN * TR * TC);
	/*for (int i = 0; i < 1024; i++)
		printf("%.17f\n", input[i]);*/

	int tn, tr, tc;
	int input_tmp_offset = 0;
	for (tn = 0; tn < TN; tn++)
		for (tr = 0; tr < TR + 4; tr++)
			for (tc = 0; tc < TC + 4; tc++)
			{
				if ((tr == 0) || (tc == 0) || (tr == TR + 2) || (tc == TC + 2) || (tr == 1) || (tc == 1) || (tr == TR + 3) || (tc == TC + 3))
					input_buffer[tn][tr][tc] = 0;
				else
				{
					input_buffer[tn][tr][tc] = input_tmp_buffer[input_tmp_offset];
					/*if(r==32)
						printf("%.17f\n", input_buffer[tn][tr][tc]);*/
					input_tmp_offset++;
				}
			}

	/*for(int i=0; i < 5; i++)
		for (int j = 0; j < 5; j++)
		{
			printf("%.17f\n", input_buffer[0][i][j]);
		}*/
}

/*void input_load_fc(float* input, float input_buffer[Tn][Tr + 4][Tc + 4], int offset, bool enable)
{
	if (!enable)
		return;

	for (int i = 0; i < 4; i++)
		input_buffer[i][0][0] = input[i + offset];
}*/

void weight_load(float* weight, float weight_buffer[Tn][Tm][K][K],
	int TN, int TM, int k, int offset, bool enable)
{
	if (enable)
		return;
	
	//	static float weight_tmp_buffer[n*m*k*k];

	float* weight_tmp_buffer = (float*)calloc(TN * TM * k * k, sizeof(float));

	memcpy((float*)weight_tmp_buffer, (float*)(weight + offset), TN * TM * k * k * sizeof(float));

	int tm, tn, kr, kc;
	int weight_tmp_offset = 0;
	for (tm = 0; tm < TM; tm++)
		for (tn = 0; tn < TN; tn++)
			for (kr = 0; kr < k; kr++)
				for (kc = 0; kc < k; kc++)
				{
					weight_buffer[tn][tm][kr][kc] = weight_tmp_buffer[weight_tmp_offset];
					weight_tmp_offset++;
				}

	/*for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
		{
			printf("%.17f\n", weight_buffer[0][0][i][j]);
		}*/
}

void weight_load_fc(float weight_buffer_fc[8 * 8 * 64], float* Weight, int offset, bool enable)
{
	if (!enable)
		return;

	memcpy((float*)weight_buffer_fc, (float*)Weight + offset, 8 * 8 * 64 * sizeof(float));
}

void output_write(float output_buffer[Tm][Tr][Tc], float* Output, int TR, int TC, int TM,
	int offset, bool write_flag, bool half)
{
	if (half)
		return;
	else
	{
		TR = TR / 2;
		TC = TC / 2;
		offset = offset / 4;
	}

	if (!write_flag)
		return;

	//	static float output_tmp_buffer[m*r*c];

	float* output_tmp_buffer = (float*)calloc(TM * TR * TC, sizeof(float));

	int tm, tr, tc;
	int output_tmp_offset = 0;
	for (tm = 0; tm < TM; tm++)
		for (tr = 0; tr < TR; tr++)
			for (tc = 0; tc < TC; tc++)
			{
				output_tmp_buffer[output_tmp_offset] = output_buffer[tm][tr][tc];
				output_tmp_offset++;
			}

	memcpy((float*)(Output + offset), (float*)output_tmp_buffer, TM * TR * TC * sizeof(float));
}

void output_write_reorg(float output_buffer[Tm][Tr][Tc], float* Output, int TR, int TC, int TM,
	int offset, bool write_flag, bool half)
{
	if (half)
		return;
	else
	{
		TR = TR / 2;
		TC = TC / 2;
		offset = offset / 4;
	}

	if (!write_flag)
		return;

	//	static float output_tmp_buffer[m*r*c];

	float* output_tmp_buffer = (float*)calloc(TM * TR * TC, sizeof(float));

	int tm, tr, tc;
	int output_tmp_offset = 0;
	for (tr = 0; tr < TR; tr++)
		for (tc = 0; tc < TC; tc++)
			for (tm = 0; tm < TM; tm++)
			{
				output_tmp_buffer[output_tmp_offset] = output_buffer[tm][tr][tc];
				output_tmp_offset++;
			}

	memcpy((float*)(Output + offset), (float*)output_tmp_buffer, TM * TR * TC * sizeof(float));
}

void output_write_fc(float output_buffer[1024], float* Output, bool enable)
{
	if (!enable)
		return;

	memcpy((float*)Output, (float*)output_buffer, 1024 * sizeof(float));
}

void relu(float input[Tm][Tr][Tc], int TM, int TR, int TC, bool enable)
{
	if (!enable)
		return;

	int tm, tr, tc;

	for (tm = 0; tm < TM; tm++)
		for (tr = 0; tr < TR; tr++)
			for (tc = 0; tc < TC; tc++)
			{
				if (input[tm][tr][tc] < 0)
					input[tm][tr][tc] = 0;
				else
					input[tm][tr][tc] = input[tm][tr][tc];
				/*if (TR==16)
					printf("relu: %.17f\n", input[tm][tr][tc]);*/
			}
	/*if (TR == 16)
		printf("0  0  0:%.17f,\n 0  2  2:%.17f,\n 0 10 10:%.17f\n", input[0][0][0], input[0][2][2], input[0][10][10]);*/
}

void relu_fc(float input[1024], bool enable)
{
	if (!enable)
		return;

	for (int i = 0; i < 1024; i++)
	{
		if (input[i] < 0)
			input[i] = 0;
		else
			input[i] = input[i];
	}
}

void copy_local_beta(float beta_buffer[MAX_BETA_LENGTH], float local_beta_buffer[MAX_BETA_LENGTH], const int TM, int m)
{
	int offset;
	int tm;
	for (tm = 0, offset = m; tm < TM; tm++)
	{
		local_beta_buffer[tm] = beta_buffer[offset];
		offset++;
	}
}

void conv(float input_buffer[Tn][Tr + 4][Tc + 4], float output_buffer[Tm][Tr][Tc],
	float weight_buffer[Tn][Tm][K][K], float beta_buffer[MAX_BETA_LENGTH],
	const int Kernel_size, const int Kernel_stride, int TM_offset,
	const int TN, const int TM, const int TR, const int TC, bool enable, const int n)
{

	float local_beta_buffer[Tm];

	/*if(TR==16)
		printf("bias: %.17f\n", local_beta_buffer[0]);*/

	if (enable)
	{
		return;
	}

	copy_local_beta(beta_buffer, local_beta_buffer, TM, TM_offset);

	float partial_mul[Tn][Tm];

	int i, j, tr, tc;
	int tm, tn;

	for (i = 0; i < Kernel_size; i++)
		for (j = 0; j < Kernel_size; j++)
			for (tr = 0; tr < TR; tr++)
				for (tc = 0; tc < TC; tc++)
				{
					for (tm = 0; tm < TM; tm++)
					{
						float tmp_add_result;
						if (i == 0 && j == 0 && n == 0)
						{
							tmp_add_result = local_beta_buffer[tm];
						}
						else
							tmp_add_result = output_buffer[tm][tr][tc];

						float partial_sum = 0;

						for (tn = 0; tn < TN; tn++)
						{
							partial_mul[tn][tm] = input_buffer[tn][Kernel_stride * tr + i][Kernel_stride * tc + j] * weight_buffer[tn][tm][i][j];
							partial_sum += partial_mul[tn][tm];
						}

						output_buffer[tm][tr][tc] = partial_sum + tmp_add_result;
						//printf("%.17f\n", output_buffer[tm][tr][tc]);
					}
				}

}

void fc(float input_buffer[8 * 8 * 64], float output_buffer[1024], float beta_buffer[1024],
	float weight_buffer[8 * 8 * 64], bool enable, int m, int n)
{
	if (!enable)
		return;

	float partial_mul;

	for (int i = 0; i < 8 * 8 * 64; i++)
	{
		float temp_add_result;
		if (i == 0 && n == 0)
			temp_add_result = beta_buffer[m];
		else
			temp_add_result = output_buffer[m];

		partial_mul = input_buffer[i] * weight_buffer[i];

		output_buffer[m] = partial_mul + temp_add_result;
	}
}

void pool(float Input[Tm][Tr][Tc], float Output[Tm][Tr][Tc],
	const int Kernel_size, const int Kernel_stride,
	const int TM, const int TR, const int TC, bool enable)
{

	if (!enable)
		return;

	int tm, tr, tc;

	float tmp0, tmp1;
	//printf("%.17f\n", Input[0][10][10]);

	for (tr = 0; tr < TR; tr++)
		for (tc = 0; tc < TC; tc++)
			for (tm = 0; tm < TM; tm++)
			{
				if (Input[tm][Kernel_stride * tr][Kernel_stride * tc] > Input[tm][Kernel_stride * tr][Kernel_stride * tc + 1])
					tmp0 = Input[tm][Kernel_stride * tr][Kernel_stride * tc];
				else
					tmp0 = Input[tm][Kernel_stride * tr][Kernel_stride * tc + 1];

				if (Input[tm][Kernel_stride * tr + 1][Kernel_stride * tc] > Input[tm][Kernel_stride * tr + 1][Kernel_stride * tc + 1])
					tmp1 = Input[tm][Kernel_stride * tr + 1][Kernel_stride * tc];
				else
					tmp1 = Input[tm][Kernel_stride * tr + 1][Kernel_stride * tc + 1];

				if (tmp0 > tmp1)
					Output[tm][tr][tc] = tmp0;
				else
					Output[tm][tr][tc] = tmp1;
			}
}

void detection_acc(float* Input, float* Output, float* Weight, float* Beta, const int InFM_num, const int OutFM_num,
	const int Kernel_size, const int Kernel_stride, const int TM, const int TN, const int TR, const int TC,
	const int mLoops, const int nLoops, const int LayerType)
{
	static float input_buffer[Tn][Tr + 4][Tc + 4];
	static float input_buffer_fc[8 * 8 * 64];
	static float output_buffer[Tm][Tr][Tc];
	static float output_buffer_fc[1024];
	static float weight_buffer[Tn][Tm][K][K];
	static float weight_buffer_fc[8 * 8 * 64];
	static float beta_buffer[MAX_BETA_LENGTH];

	memcpy(beta_buffer, Beta, OutFM_num * sizeof(float));

	if (LayerType == 1)
		memcpy((float*)input_buffer_fc, (float*)Input, 4096 * sizeof(float));

	int m, n;
	for (m = 0; m < mLoops; m++)
	{
		for (n = 0; n < nLoops; n++)
		{
			input_load_conv(Input, input_buffer, TR, TC, TN, n * TN * TR * TC, LayerType);
			//input_load_fc(input_buffer_fc, input_buffer, m * 4 * nLoops + n * 4, LayerType);
			weight_load(Weight, weight_buffer, TN, TM, Kernel_size, m * TN * TM * nLoops * Kernel_size * Kernel_size + n * TM * TN * Kernel_size * Kernel_size, LayerType);
			weight_load_fc(weight_buffer_fc, Weight, m * 8 * 8 * 64, LayerType);
			conv(input_buffer, output_buffer, weight_buffer, beta_buffer, Kernel_size, Kernel_stride, m * TM, TN, TM, TR, TC, LayerType, n);
			fc(input_buffer_fc, output_buffer_fc, beta_buffer, weight_buffer_fc, LayerType, m, n);
		}
		relu(output_buffer, TM, TR, TC, n==nLoops);
		pool(output_buffer, output_buffer, 2, 2, TM, TR / 2, TC / 2, (!LayerType)&(n==nLoops));
		if (TR == 32)
			output_write(output_buffer, Output, TR, TC, TM, m * TM * TR * TC, n == nLoops, LayerType);
		else
			output_write(output_buffer, Output, TR, TC, TM, m * TM * TR * TC, n == nLoops, LayerType);
	}
	relu_fc(output_buffer_fc, LayerType);
	output_write_fc(output_buffer_fc, Output, LayerType);
}
void file_error(char* s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

/*image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}


image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}

image load_image_stb(char* filename, int channels)
{
    int w, h, c;
    unsigned char* data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + c * i + c * w * j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}*/

void calculator_ps(float* input)
{
    int weight_offset[3] = { 800, 51200, 4194304 };
    int beta_offset[3] = { 32, 64, 1024 };

    float* Weight_buf = (float*)calloc(4260640, sizeof(float));
    float* Beta_buf = (float*)calloc(1134, sizeof(float));

    FILE* fp_w = fopen("./merged_float32_weights.bin", "rb");
    if (!fp_w) file_error("weight.bin");

    FILE* fp_b = fopen("./merged_float32_biases.bin", "rb");
    if (!fp_b) file_error("bias.bin");

    fread(Weight_buf, sizeof(float), 4260640, fp_w);
    fread(Beta_buf, sizeof(float), 1134, fp_b);

    /*for (int i = 0; i < 1134; i++)
        printf("bias[%d]:%.17f\n",i, Beta_buf[i]);*/

    fclose(fp_w);
    fclose(fp_b);

#define MEM_LEN (16*16*32+8*8*64)
    float* Memory_buf = (float*)calloc(MEM_LEN + 1024 * 2, sizeof(float));
    float* Memory_top = Memory_buf + 1024;
    float* Memory_bottom = Memory_top + MEM_LEN;
    memcpy(Memory_top, input, 32 * 32 * 1 * sizeof(float));

    float* in_ptr[4];
    float* out_ptr[4];

    in_ptr[0] = Memory_top;
    out_ptr[0] = Memory_bottom - 16 * 16 * 32;

    in_ptr[1] = out_ptr[0];
    out_ptr[1] = Memory_top;

    in_ptr[2] = out_ptr[1];
    out_ptr[2] = Memory_bottom - 1024;

    in_ptr[3] = out_ptr[2];
    out_ptr[3] = Memory_top;

    int i;
    int woffset = 0;
    int boffset = 0;
    int TR, TC, TM, TN;
    int mLoops, nLoops;

    for (i = 0; i < 4; ++i)
    {
        if (i == 0)
        {
            printf("Conv0\n");

            TR = 32;
            TC = 32;

            TM = 32;
            TN = 1;

            mLoops = 1;
            nLoops = 1;

            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                1, 32, 5, 1, TM, TN, TR, TC, mLoops, nLoops, 0);

            woffset += weight_offset[i];
            boffset += beta_offset[i];
        }
        else if (i == 1)
        {
            printf("Conv2\n");

            TR = 16;
            TC = 16;

            TM = 32;
            TN = 4;

            mLoops = 2;
            nLoops = 8;

            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                32, 64, 5, 1, TM, TN, TR, TC, mLoops, nLoops, 0);

            woffset += weight_offset[i];
            boffset += beta_offset[i];

            /*for (int j = 8092; j < 8192; j++)
                printf("%.17f\n", in_ptr[i][j]);

            printf("\n");*/
            //printf("\n");

            /*for (int j = 3996; j < 4096; j++)
            {
                printf("%.17f\n", out_ptr[i][j]);
            }*/
            //printf("%.17f", out_ptr[i][0]);
        }
        else if (i == 2)
        {
            printf("FC4\n");

            float reorg_out[64][8][8];

            for (int m = 0; m < 64; m++)
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        reorg_out[m][r][c] = in_ptr[2][m * 64 + r * 8 + c];
                        //printf("reo1:%d\n", m * 64 + r * 8 + c);
                    }

            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    for (int m = 0; m < 64; m++)
                    {
                        in_ptr[2][r * 512 + c * 64 + m] = reorg_out[m][r][c];
                        //printf("reo2:%d\n", r * 512 + c * 64 + m);
                    }
            /*printf("\n");
            for (int j = 3996; j < 4096; j++)
            {
                printf("%.17f\n", in_ptr[i][j]);
            }
            printf("\n");*/
            
            TR = 1;
            TC = 1;

            TM = 32;
            TN = 4;

            mLoops = 1024;
            nLoops = 1;

            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                8 * 8 * 64, 1024, 1, 1, TM, TN, TR, TC, mLoops, nLoops, 1);

            woffset += weight_offset[i];
            boffset += beta_offset[i];

            /*for (int j = 0; j < 1024; j++)
                printf("%d: %.17f\n", j, out_ptr[i][j]);*/

            /*float* in_buf = (float*)calloc(4096, sizeof(float));

            FILE* fp_w = fopen("fc_0_0_input.bin", "rb");
            if (!fp_w) file_error("fc_0_0_input.bin");

            fread(in_buf, sizeof(float), 4096, fp_w);

            fclose(fp_w);*/

            /*int m, n;
            for (m = 0; m < 1024; m++)
            {
                for (n = 0; n < 4096; n++)
                {
                    float tmp_add_result;

                    if (n == 0)
                        tmp_add_result = Beta_buf[96 + m];
                    else
                        tmp_add_result = out_ptr[2][m];

                    float partial_mul = in_ptr[2][n] * Weight_buf[52000 + m * 4096 + n];
                    if (m == 0 && n == 1024)
                        printf("input:%.17f\nweight:%.17f\nbias:%.17f\n", in_ptr[2][n], Weight_buf[52000 + m * 4096 + n], Beta_buf[96 + m]);

                    out_ptr[2][m] = partial_mul + tmp_add_result;
                    if (m == 0 && n == 1024)
                        printf("out[%d]:%.17f\n", n, out_ptr[2][0]);
                }
                printf("%.17f\n", out_ptr[2][m]);
            }*/
        }
        else if (i == 3)
        {
            printf("FC5\n");

            int m, n;
            for (m = 0; m < 14; m++)
            {
                for (n = 0; n < 1024; n++)
                {
                    float tmp_add_result;

                    if (n == 0)
                        tmp_add_result = Beta_buf[1120 + m];
                    else
                        tmp_add_result = out_ptr[3][m];

                    float partial_mul = in_ptr[3][n] * Weight_buf[4246304 + m * 1024 + n];

                    out_ptr[3][m] = partial_mul + tmp_add_result;
                }
                printf("%d: %.17f\n", m, out_ptr[3][m]);
            }
        }
    }

    free(Memory_buf);
    free(Weight_buf);
    free(Beta_buf);
}

int main()
{
    printf("*****Handwritten Mathematical Calculator Test Begin!*****\n");

    //   char* image_path = "./test.jpeg";

    //   image im = load_image_stb(image_path, 1);
    //   printf("Input img:%s\n w=%d,h=%d,c=%d\n", image_path, im.w, im.h, im.c);

    float* Input_buf = (float*)calloc(1024, sizeof(float));

    FILE* fp_i = fopen("./input2.bin", "rb");
    if (!fp_i) file_error("input2.bin");

    fread(Input_buf, sizeof(float), 1024, fp_i);

    //fclose(fp_i);

    //    float *X = im.data;

    time_t first, second;

    first = time(NULL);
    calculator_ps(Input_buf);
    second = time(NULL);
    printf("Predicted in %f seconds.\n", difftime(second, first));
}