// Copyright CERN. This software is distributed under the terms of the GNU
// General Public License v3 (GPL Version 3).
//
// See http://www.gnu.org/licenses/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliONNXInterface.cxx
/// \author fabio.catalano@cern.ch, maximiliano.puccio@cern.ch

#include "AliONNXInterface.h"
#include <iostream>
#include <cassert>
#include <TSystem.h>
#include <TGrid.h>
#include <TDirectory.h>
#include <TFile.h>

AliONNXInterface::AliONNXInterface(std::string name, bool debug) :
  fInterfaceName{name},
  fEnv{nullptr},
  fSessionOptions{nullptr},
  fSession{nullptr},
  fInputNodeName{},
  fInputNodeDim{},
  fOutputNodeName{},
  fAllocatorInfo{nullptr}
{
  OrtLoggingLevel log_level;
  const char* env_name;
  if(debug) {
    log_level = ORT_LOGGING_LEVEL_VERBOSE;
    env_name = "debug";
  }
  else {
    log_level = ORT_LOGGING_LEVEL_WARNING;
    env_name = fInterfaceName.c_str();
  }

  checkStatus(OrtCreateEnv(log_level, env_name, &fEnv));
  fSessionOptions = OrtCreateSessionOptions();
  OrtSetSessionThreadPoolSize(fSessionOptions, 1);
  OrtSetSessionGraphOptimizationLevel(fSessionOptions, 1);
}

AliONNXInterface::~AliONNXInterface(){
  OrtReleaseAllocatorInfo(fAllocatorInfo);
  OrtReleaseSession(fSession);
  OrtReleaseSessionOptions(fSessionOptions);
  OrtReleaseEnv(fEnv);
}

bool AliONNXInterface::LoadXGBoostModel(std::string path, int size) {
  if (path.empty()) {
    std::cerr << "Invalid empty model path string!" << std::endl;
    return false;
  }
  path = GetFile(path);
  OrtAllocator* allocator;
  OrtStatus* status;
  size_t num_input_nodes;
  checkStatus(OrtCreateSession(fEnv, path.c_str(), fSessionOptions, &fSession));
  checkStatus(OrtCreateDefaultAllocator(&allocator));
  status = OrtSessionGetInputCount(fSession, &num_input_nodes);
  if(num_input_nodes != 1) {
    std::cerr << "Case with more than one input node not impelemented!" << std::endl;
    return false;
  }
  char* input_name;
  status = OrtSessionGetInputName(fSession, 0, allocator, &input_name);
  fInputNodeName = input_name;
  OrtTypeInfo* typeinfo;
  status = OrtSessionGetInputTypeInfo(fSession, 0, &typeinfo);
  const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
  size_t num_dims = OrtGetNumOfDimensions(tensor_info);
  fInputNodeDim.resize(num_dims);
  OrtGetDimensions(tensor_info, (int64_t*)fInputNodeDim.data(), num_dims);
  if( size != fInputNodeDim[1]) {
    std::cerr << "The input size entered doesn't match the model one!" << std::endl;
    return false;
  }
  fOutputNodeName = "probabilities";
  checkStatus(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &fAllocatorInfo));

  OrtReleaseTypeInfo(typeinfo);
  OrtReleaseStatus(status);
  OrtReleaseAllocator(allocator);

  return true;
}

float AliONNXInterface::Predict(float *features, int size) {
  float output = 0.f;
  // create input tensor object from data values
  OrtValue* input_tensor = NULL;
  checkStatus(OrtCreateTensorWithDataAsOrtValue(fAllocatorInfo, features, size * sizeof(float), 
                                                fInputNodeDim.data(),fInputNodeDim.size(),
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  assert(OrtIsTensor(input_tensor));
  // score model & input tensor, get back output tensor
  OrtValue* output_tensor = NULL;
  checkStatus(OrtRun(fSession, NULL, &fInputNodeName, (const OrtValue* const*)&input_tensor, 1,
                     &fOutputNodeName, 1, &output_tensor));
  assert(OrtIsTensor(output_tensor));
  // Get pointer to output tensor float values
  float* floatarr;
  checkStatus(OrtGetTensorMutableData(output_tensor, (void**)&floatarr));
  output = floatarr[1];
  OrtReleaseValue(output_tensor);
  OrtReleaseValue(input_tensor);

  return output;
}

void AliONNXInterface::checkStatus(OrtStatus* onnx_status) {
  if (onnx_status != NULL) {
    const char* msg = OrtGetErrorMessage(onnx_status);
    std::cerr << msg << std::endl;
    OrtReleaseStatus(onnx_status);
    exit(1);
  }
}

std::string AliONNXInterface::GetFile(const std::string path) {
  if(path.find("alien:") != std::string::npos) {
    size_t pos = path.find_last_of("/") + 1;
    std::string model_name = path.substr(pos);
    if(gGrid == nullptr) {
      TGrid::Connect("alien://");
      if(gGrid == nullptr){
        std::cerr << "Connection to GRID not established!" << std::endl;
      }
    }
    std::string new_path = gSystem->pwd() + std::string("/") + model_name.data();
    const char *old_root_dir = gDirectory->GetPath();
    bool cp_status = TFile::Cp(path.data(), new_path.data());
    gDirectory->cd(old_root_dir);
    if(!cp_status){
      std::cerr << "Error in coping file from Alien!\n";
      return std::string();
    }
    return new_path;
  } else {
    return path;
  }
}