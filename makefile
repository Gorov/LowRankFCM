CC = g++
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops #-Wno-unused-result
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops #-Wno-unused-result

OBJS = BaseComponentModel.o FeatureEmbeddingModel.o FullFctModel.o LabelEmbeddingModel.o LrFcemModel.o LrFcemModelLabel.o LrFcemMultitask.o LrRankingModel.o LrTensorBigramModel.o LrTensorModelBasic.o LrTensorModel.o LrTensorSparseModel.o LrTensorTuckerModel.o LrTuckerRankingModel.o RunBigramRanking.o RunRankingModel.o WordEmbeddingModel.o FeatureFactory.o RunCombinedRanker.o

all: run run_brown

%.o : %.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

run : run.cpp $(OBJS)
	$(CC) run.cpp $(OBJS) -o run $(CFLAGS)

run_brown : run_brown.cpp $(OBJS)
	$(CC) run_brown.cpp $(OBJS) -o run_brown $(CFLAGS)

clean:
	rm -rf run run_brown *.o
