/*
* Copyright 2017 [Zhouyuan Huo]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/

#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
 protected:
  std::vector<double> model_copy;
  Gradient * full_gradient;
  //double momentum_rate;

    virtual void ApplyGradient(Gradient *gradient, double learning_rate) override {
        //printf("here");
        std::vector<double> &model_data = model->ModelData();
        int para_size = model->NumParameters();
        for (int i=0; i<para_size;i++) {
            model_data[i] -= learning_rate * gradient->coeffs[i];
        }

        double epo = (2*learning_rate) / (FLAGS_learning_rate) -2;
        double momentum_rate = 2/(epo + 2);

        std::vector<double> &model_momentum = model->MomentumData();
        for (int i=0; i<para_size; i++) {
            model_momentum[i] = model_copy[i] + momentum_rate * (model_data[i] - model_copy[i]);
        }
        //compute the distance between x and y
        //bindouble d = 0;
       // double coor = 0;
        //lost+foundr (int i = 0; i < model_data.size(); i++){
            //coor = model_momentum[i] - model_data[i];
           /// d += coor * coor;
       // }
       //  printf("Here is the information: %f\n", d);
       // d = 0;

    }


  virtual void PrepareGradient(Datapoint *datapoints, Gradient *gradient, const std::vector<int> &left_right) override {
    Gradient * prev_gradient = new Gradient();
	//std::vector<double> &cur_model = model->ModelData();
    std::vector<double> &cur_model = model->MomentumData();

	model->PrecomputeCoefficients(datapoints, gradient, cur_model, left_right);
	model->ComputeL2Gradient(gradient, cur_model);

	model->PrecomputeCoefficients(datapoints, prev_gradient, model_copy, left_right);
	model->ComputeL2Gradient(prev_gradient, model_copy);

	for (int i = 0; i < model->NumParameters(); i++){
	  gradient->coeffs[i] += - prev_gradient->coeffs[i] + full_gradient->coeffs[i];
	}

	delete prev_gradient;
  }

  // compute full gradient and store model
  void ModelCopy() {
	int worker_num = 0, master_num = 0;
	Gradient * gradient = new Gradient();
	gradient->coeffs.resize(model->NumParameters(), 0);

	full_gradient->coeffs = gradient->coeffs;
	//std::vector<double> &cur_model = model->ModelData();
    std::vector<double> &cur_model = model->MomentumData();

	model_copy = cur_model;

	if (model->taskid == 0) {
	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&gradient->coeffs[0], &full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  for(int i=0; i < model->NumParameters(); i++) {
		full_gradient->coeffs[i] /= master_num;
	  }
	}
	else {
	  worker_num = datapoints->GetSize();
	  std::vector<int> left_right(2, 0);
	  left_right[1] = worker_num;

	  model->PrecomputeCoefficients(datapoints, gradient, cur_model, left_right);
	  model->ComputeL2Gradient(gradient, cur_model);

	  for(int i=0; i < model->NumParameters(); i++) {
		gradient->coeffs[i] *= worker_num;
	  }

	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&gradient->coeffs[0], &full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(&full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	delete gradient;

    //std::vector<double> &cur_y = model->ModelData();

    //double epo = 2*learning_rate / FLAGS_learning_rate -2;
   // double momentum_rate = 2/ (epo + 2);

    //for (int i=0; i< model->NumParameters(); i++) {
      //cur_model[i] = momentum_rate * cur_y[i] + (1-momentum_rate) * model_copy[i];
    //}
  }

 public:
  SVRGUpdater(Model *model, Datapoint *datapoints) : Updater(model, datapoints) {
	full_gradient = new Gradient();
    model_copy.resize(FLAGS_d1,0);
   // momentum_rate = 0；
  }

  void UpdateCurModel(double learning_rate) override {
    std::vector<double>& cur_model = model->MomentumData();
    std::vector<double>& cur_y = model->ModelData();

    double epo = 2*learning_rate /FLAGS_learning_rate - 2;
    double momentum_rate = 2/(epo + 2);
    for (int i=0; i<model->NumParameters(); i++) {
     cur_model[i] = momentum_rate * cur_y[i] + (1-momentum_rate) * model_copy[i];
    }
  }

  virtual void EpochBegin() override {
	Updater::EpochBegin();
	ModelCopy();
  }

  ~SVRGUpdater() {
	delete full_gradient;
  }
};

#endif
