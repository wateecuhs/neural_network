#include "displays.h"
#include "network.h"
#include "activation.h"
#include "loss.h"
#include "displays.h"
#include "dataset.h"


/*
 DONT LOOK AT THIS CODE IT'S DOGSHIT IM JUST TRYING STUFF
 DONT LOOK AT THIS CODE IT'S DOGSHIT IM JUST TRYING STUFF
 DONT LOOK AT THIS CODE IT'S DOGSHIT IM JUST TRYING STUFF
 DONT LOOK AT THIS CODE IT'S DOGSHIT IM JUST TRYING STUFF
 DONT LOOK AT THIS CODE IT'S DOGSHIT IM JUST TRYING STUFF
 */


/* 
const int FPS = 60;
const int FRAME_DELAY = 1000 / FPS;

void draw_default_frame(Display *display)
{
    SDL_SetRenderDrawColor(display->renderer, 0, 0, 0, 255);
    SDL_RenderClear(display->renderer);
    SDL_SetRenderDrawColor(display->renderer, 120, 120, 120, 255);
    SDL_SetRenderDrawColor(display->renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(display->renderer, &(SDL_Rect){10, 10, 620, 460});
    SDL_RenderDrawRect(display->renderer, &(SDL_Rect){38, 38, 284, 284});
    // SDL_RenderFillRect(display->renderer, &(SDL_Rect){40, 40, 280, 280});
    SDL_SetRenderDrawColor(display->renderer, 255, 255, 255, 255);
    for (int i = 0; i < 784; i++) {
        if (display->input[i] == 1) {
            SDL_RenderFillRect(display->renderer, &(SDL_Rect){40 + (i % 28) * 10, 40 + (i / 28) * 10, 10, 10});
        }
    }
    SDL_RenderPresent(display->renderer);
}

void display_loop(Display *display)
{
    uint32_t frameStart;
    SDL_Event event;
    int frameTime;
    int quit = 0;
    int x, y;

    display-> mask = 0;
    bzero(display->input, 784);

    while (!quit) {
        frameStart = SDL_GetTicks();
        draw_default_frame(display);
        if (display-> mask & 1 << 0) {
            SDL_GetMouseState(&x, &y);
            if (!(x < 40 || y < 40 || x > 319 || y > 319)) {
                x = (x - 40) / 10;
                y = (y - 40) / 10;
                display->input[y * 28 + x] = 1;
            }
        }
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = 1;
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT)
                display-> mask ^= 1 << 0;
            if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT)
                display-> mask ^= 1 << 0;
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE)
                bzero(display->input, 784);
        }
        frameTime = SDL_GetTicks() - frameStart;
        if (frameTime < FRAME_DELAY) {
            SDL_Delay(FRAME_DELAY - frameTime);
        }
    }
}

void progress_bar(Display *display)
{
    Network	*nn;
    Dataset dataset = unpack_mnist();
    double *d_loss = NULL;

    nn = malloc(sizeof(Network));
    if (!nn)
        exit(EXIT_FAILURE);

    init_network(nn);
    add_layer(nn, 784, RELU);
    add_layer(nn, 64, RELU);
    add_layer(nn, 32, RELU);
    add_layer(nn, 10, SOFTMAX);

    double *inputs = malloc(784 * sizeof(double));
    double expected[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 60000; i++)
    {
        memcpy(inputs, &dataset.inputs[i * 784], 784 * sizeof(double));
        nn_forward(nn, inputs, 784);
        bzero(expected, 10 * sizeof(double));
        expected[dataset.targets[i]] = 1;
        d_loss = d_loss_softmax_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, dataset.targets[i]);
        nn_backward(nn, d_loss);
        free(d_loss);
        for (int i = 0; i < nn->nb_layers; i++)
            update_parameters(&nn->layers[i], 0.001);
        SDL_SetRenderDrawColor(display->renderer, 0, 0, 0, 255);
        SDL_RenderClear(display->renderer);
        SDL_SetRenderDrawColor(display->renderer, 255, 255, 255, 255);
        SDL_RenderDrawRect(display->renderer, &(SDL_Rect){120, 240, 400 ,30});
        SDL_RenderFillRect(display->renderer, &(SDL_Rect){120, 240, 400 * (i + 1) / 60000, 30});
        SDL_RenderPresent(display->renderer);
    }
    free(inputs);
    free_network(nn);
}

void init_display(void)
{
    Display display;
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    display.window = SDL_CreateWindow(
        "Local SDL Test",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        640,
        480,
        SDL_WINDOW_SHOWN
    );

    if (!display.window) {
        printf("Failed to create window: %s\n", SDL_GetError());
        SDL_Quit();
        exit(EXIT_FAILURE);
    }

    display.renderer = SDL_CreateRenderer(display.window, -1, SDL_RENDERER_ACCELERATED);
    if (!display.renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(display.window);
        SDL_Quit();
        exit(EXIT_FAILURE);
    }
    progress_bar(&display);
    display_loop(&display);

    SDL_DestroyRenderer(display.renderer);
    SDL_DestroyWindow(display.window);
    SDL_Quit();
} */