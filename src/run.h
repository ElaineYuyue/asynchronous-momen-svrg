#include <iostream>
#include "defines.h"

template<class MODEL_CLASS = LSL2L1Model, class DATAPOINT_CLASS = ARMADatapoint, class CUSTOM_UPDATER=SVRGUpdater>
TrainStatistics RunOnce(int taskid) {
    // Initialize model and datapoints.
  Model *model = new MODEL_CLASS(taskid);

  Datapoint *datapoints = new DATAPOINT_CLASS(FLAGS_data_file, taskid);


  if (FLAGS_multi_class_trace)
	datapoints->OnehotEncoding(FLAGS_d2);
  model->SetUp(datapoints); //确定label正负的个数

  Updater *updater = NULL;
  if (FLAGS_svrg) {
	  updater = new SVRGUpdater(model, datapoints);
  }
  else if (FLAGS_sgd) {
	updater = new SGDUpdater(model, datapoints);
  }
  else if (FLAGS_dfsdca) {
	updater = new DFSDCAUpdater(model, datapoints);
  }
  else {
	updater = new CUSTOM_UPDATER(model, datapoints);
  }

  // Create trainer depending on flag.
  Trainer *trainer = NULL;
  if (!FLAGS_decouple){
    if (taskid == 0) {
      //printf("server %d\n", datapoints->GetSize());
      trainer = new ServerTrainer(model, datapoints);
    }
    else {
      trainer = new WorkerTrainer(model, datapoints);
    }
  }
  else{
    if (taskid == 0) {
      trainer = new DecoupledServerTrainer(model, datapoints);
    }
    else {
      trainer = new DecoupledWorkerTrainer(model, datapoints);
    }
  }

  TrainStatistics stats = trainer->Train(model, datapoints, updater);

  // Delete trainer.
  delete trainer;

  // Delete model and datapoints.
  delete model;
  delete datapoints;

  // Delete updater.
  delete updater;

  return stats;
}

template<class MODEL_CLASS = LSL2L1Model, class DATAPOINT_CLASS = ARMADatapoint, class CUSTOM_UPDATER=SVRGUpdater>
void Run(int taskid) {
  TrainStatistics stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS, CUSTOM_UPDATER>(taskid);
}
