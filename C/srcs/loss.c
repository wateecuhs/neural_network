#include <math.h>
#include <stdio.h>

double cce_loss(double *pred, int pred_len, double *expected)
{
	double loss = 0;

	for (int i = 0; i < pred_len; i++)
	{
		if (pred[i] < 1e-7)
			pred[i] = 1e-7;
		else if (pred[i] > 1-1e-7)
			pred[i] = 1-1e-7;
		if (expected[i] != 0)
			loss += -log(pred[i] * expected[i]);
		// printf("singular loss for %d of value %f, expecting %f: %f\n", i, pred[i], expected[i], -log(pred[i] * expected[i]));
	}
	return (loss);
}