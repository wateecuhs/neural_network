NAME				=	OCR

include config/srcs.mk
SRC_PATH			=	srcs/
SDL_PATH			 =	lib
DIR_BUILD			=	.build/
OBJS				=	$(patsubst %.c, $(DIR_BUILD)%.o, $(SRCS))
OBJS_TEST			=	$(patsubst %.c, $(DIR_BUILD)%.o, $(TEST))
DEPS				=	$(patsubst %.c, $(DIR_BUILD)%.d, $(SRCS))

DEPS_FLAGS			=	-MMD -MP
LIB_FLAGS			=	-lm #-L$(SDL_PATH)/lib -lSDL2 -lm -lpthread -ldl -Wl,-rpath,$(SDL_PATH)/lib -D_REENTRANT
BASE_CFLAGS			=	-g3 -Wall -Wextra -Werror
BASE_DEBUG_CFLAGS	=	-g3
DEBUG_CLFAGS		=	$(BASE_DEBUG_CFLAGS) -fsanitize=address
FLAGS				=	$(BASE_CFLAGS)

RM					=	rm -rf


DIR_INCS =\
	includes/ \
	$(SDL_PATH)/include/SDL2 \

INCLUDES =\
	$(addprefix -I , $(DIR_INCS))


.PHONY:		all
all:
			$(MAKE) $(NAME)


$(NAME):	$(OBJS)
	$(CC) $(FLAGS) $(INCLUDES) $(OBJS) -o $(NAME) $(LIB_FLAGS)

.PHONY:	clean
clean:
			$(RM) $(DIR_BUILD)
	
.PHONY:	fclean
fclean:	clean
			$(RM) $(NAME)

.PHONY:	debug
debug:	fclean
			$(MAKE) -j FLAGS="$(DEBUG_CLFAGS)"

.PHONY:	re
re:		fclean
			$(MAKE) all

-include $(DEPS)
$(DIR_BUILD)%.o : $(SRC_PATH)%.c
			@mkdir -p $(shell dirname $@)
			$(CC) $(INCLUDES) -c $< -o $@ $(FLAGS) $(DEPS_FLAGS)