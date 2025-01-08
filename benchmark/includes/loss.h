#ifndef LOSS_H
# define LOSS_H

double  loss_cce(double *pred, int pred_len, double *expected);
double  *d_loss_softmax_cce(double *inputs, int nb_inputs, int true_class);

#endif