#include <math.h>
#include <stdio.h>
#include "activation.h"

double loss_cce(double *pred, int pred_len, double *expected)
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

double loss_softmax_cce(double *pred, int pred_len, double *expected)
{
	pred = softmax(pred, pred_len);
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

double *d_loss_softmax_cce(double *dvalues, int dvalues_len, int true_class)
{
	dvalues[true_class] -= 1;
	(void)dvalues_len;
	return dvalues;
}