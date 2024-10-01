#ifndef NEURON_H
# define NEURON_H

typedef struct s_neuron {
	double	*weights;
	double	*input;
	int		input_len;
	double	bias;
	double	output;
	double	error;
	double	(*activation)(double x);
	void	(*derivative)(struct s_neuron *neuron);
	void	(*update)(struct s_neuron *neuron, double lr);
} Neuron;

int		init_neuron(Neuron *neuron, int nn_prev_layer, double (*activation)(double x));
int		neuron_forward(Neuron *neuron, double *inputs, int nb_inputs);
void	free_neuron(Neuron *neuron);

#endif