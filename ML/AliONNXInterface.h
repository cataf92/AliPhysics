// Copyright CERN. This software is distributed under the terms of the GNU
// General Public License v3 (GPL Version 3).
//
// See http://www.gnu.org/licenses/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliONNXInterface.h
/// \brief Implementation of a C++ interface in AliPhysics for the application of 
///        ONNX exported models
/// \author fabio.catalano@cern.ch

#ifndef ALIONNXINTERFACE_H
#define ALIONNXINTERFACE_H

#include <string>
#include <vector>
#include "onnxruntime/core/session/onnxruntime_c_api.h"

class AliONNXInterface {
public:
  AliONNXInterface(std::string name = "", bool debug = false);
  ~AliONNXInterface();

  bool LoadXGBoostModel(std::string path, int size);

  float Predict(float *features, int size);

private:
  void checkStatus(OrtStatus* onnx_status);

  std::string fInterfaceName;
  OrtEnv* fEnv;
  OrtSessionOptions* fSessionOptions;
  OrtSession *fSession;
  const char* fInputNodeName;
  std::vector<int64_t> fInputNodeDim;  // is assumed that the model has only 1 input node
  const char* fOutputNodeName;
  OrtAllocatorInfo* fAllocatorInfo;
};

#endif
