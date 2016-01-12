SRCDIR = .
CXXFLAGS = -I${SRCDIR} -std=c++11 -Wall -Wextra -pedantic -DNDEBUG -g
CPP_FILES := $(wildcard $(SRCDIR)/*.cc)

all: fit

fit: $(CPP_FILES)
	g++ $(CXXFLAGS) $(CPP_FILES)  -o $@

clean:
	rm -f fit
