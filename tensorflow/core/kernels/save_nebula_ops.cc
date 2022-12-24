//
// Created by qinghuanrao on 9/22/2022.
//
#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include <fstream>
#include <sys/file.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
namespace tensorflow {

    namespace {

// Shared validations of the inputs to the SaveNebula and RestoreV2 ops.
        void ValidateInputs(bool is_save_op, OpKernelContext* context,
                            const Tensor& prefix, const Tensor& tensor_names,
                            const Tensor& shape_and_slices) {
            const int kFixedInputs = 3;  // Prefix, tensor names, shape_and_slices.
            const int num_tensors = static_cast<int>(tensor_names.NumElements());
            OP_REQUIRES(
                    context, prefix.NumElements() == 1,
                    errors::InvalidArgument("Input prefix should have a single element, got ",
                                            prefix.NumElements(), " instead."));
            OP_REQUIRES(context,
                        TensorShapeUtils::IsVector(tensor_names.shape()) &&
                        TensorShapeUtils::IsVector(shape_and_slices.shape()),
                        errors::InvalidArgument(
                                "Input tensor_names and shape_and_slices "
                                "should be an 1-D tensors, got ",
                                tensor_names.shape().DebugString(), " and ",
                                shape_and_slices.shape().DebugString(), " instead."));
            OP_REQUIRES(context,
                        tensor_names.NumElements() == shape_and_slices.NumElements(),
                        errors::InvalidArgument("tensor_names and shape_and_slices "
                                                "have different number of elements: ",
                                                tensor_names.NumElements(), " vs. ",
                                                shape_and_slices.NumElements()));
            OP_REQUIRES(context,
                        FastBoundsCheck(tensor_names.NumElements() + kFixedInputs,
                                        std::numeric_limits<int>::max()),
                        errors::InvalidArgument("Too many inputs to the op"));
            OP_REQUIRES(
                    context, shape_and_slices.NumElements() == num_tensors,
                    errors::InvalidArgument("Expected ", num_tensors,
                                            " elements in shapes_and_slices, but got ",
                                            context->input(2).NumElements()));
            if (is_save_op) {
                OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                            errors::InvalidArgument(
                                    "Got ", num_tensors, " tensor names but ",
                                    context->num_inputs() - kFixedInputs, " tensors."));
                OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                            errors::InvalidArgument(
                                    "Expected a total of ", num_tensors + kFixedInputs,
                                    " inputs as input #1 (which is a string "
                                    "tensor of saved names) contains ",
                                    num_tensors, " names, but received ", context->num_inputs(),
                                    " inputs"));
            }
        }

        int CopyFile(char * shm_name, char * filename){
            using namespace std;
            const char * prefix_shm = "/dev/shm/";
            std::string const& shm_path = std::string(prefix_shm) + std::string(shm_name);
            const char * command_prefix = "cp";
            std::string const& command = std::string(command_prefix) + " " + shm_path + " " + std::string(filename);
            const char *c = command.c_str();
            return system(c);
        }

        char*  random_string(std::size_t length)
        {
            const std::string CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            std::random_device random_device;
            std::mt19937 generator(random_device());
            std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

            std::string random_string;
            for (std::size_t i = 0; i < length; ++i)
            {
                random_string += CHARACTERS[distribution(generator)];
            }

            return const_cast<char *>(random_string.c_str());
        }
    }  // namespace

// Saves a list of named tensors using the tensor bundle library.
    class SaveNebula : public OpKernel {
    public:
        explicit SaveNebula(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& prefix = context->input(0);
            const Tensor& tensor_names = context->input(1);
            const Tensor& shape_and_slices = context->input(2);
            ValidateInputs(true /* is save op */, context, prefix, tensor_names,
                           shape_and_slices);

            const int kFixedInputs = 3;  // Prefix, tensor names, shape_and_slices.
            const int num_tensors = static_cast<int>(tensor_names.NumElements());
            const string& prefix_string = prefix.scalar<tstring>()();
            const auto& tensor_names_flat = tensor_names.flat<tstring>();
            const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

            BundleWriter writer(Env::Default(), prefix_string);
            OP_REQUIRES_OK(context, writer.status());
            VLOG(1) << "BundleWriter, prefix_string: " << prefix_string;
            size_t total_size = 0;
            for (int i = 0 ; i < num_tensors; ++i) {
                //const string& tensor_name = tensor_names_flat(i);
                const Tensor& atensor = context->input(i + kFixedInputs);
                total_size += writer.CalculateTensorsSize(atensor);
                if (!shape_and_slices_flat(i).empty()) {
                    //std::cout << "Nebula1 Tensor size: " << total_size << std::endl;
                    total_size += writer.CalculateTensorsSize(atensor);
                }

            }
            //std::cout << "Nebula2 Tensor size: " << total_size << std::endl;
            string data_path = DataFilename(prefix_string, 0, 1);
            const char * filename_ =const_cast<char *>(data_path.c_str());
            //std::cout << "Nebula filename_: " << filename_ << std::endl;
            std::string str;
            str = writer.md5_shm(filename_,str);
            char * shm_name = const_cast<char *>(str.c_str());
            std::cout << "Nebula shm_name: " << shm_name << std::endl;
            writer.allocate(shm_name, total_size);
            for (int i = 0; i < num_tensors; ++i) {
                const string& tensor_name = tensor_names_flat(i);
                const Tensor& tensor = context->input(i + kFixedInputs);

                if (!shape_and_slices_flat(i).empty()) {
                    const string& shape_spec = shape_and_slices_flat(i);
                    TensorShape shape;
                    TensorSlice slice(tensor.dims());
                    TensorShape slice_shape;

                    OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                            shape_spec, &shape, &slice, &slice_shape));
                    OP_REQUIRES(context, slice_shape.IsSameSize(tensor.shape()),
                                errors::InvalidArgument("Slice in shape_and_slice "
                                                        "specification does not match the "
                                                        "shape of the tensor to  save: ",
                                                        shape_spec, ", tensor: ",
                                                        tensor.shape().DebugString()));

                    OP_REQUIRES_OK(context,
                                   writer.AddSlice(tensor_name, shape, slice, tensor));
                } else {
                    OP_REQUIRES_OK(context, writer.AddShm(tensor_name, tensor, shm_name));
                }
            }
            OP_REQUIRES_OK(context, writer.Finish());
            const std::string& record = "/tmp/record";
            FILE *pFile;
            if ((pFile = fopen(record.c_str(), "a")) == NULL)
            {
                std::cout << "Failed to open record file, file path: " << std::endl;
                return;
            }
            flock(fileno(pFile), LOCK_EX | LOCK_NB);
            std::string fileData = str + "|" + data_path + "|" + std::to_string(total_size) + "\n";

            fwrite(fileData.c_str(), 1, fileData.length(), pFile);
            flock(fileno(pFile), LOCK_UN);
            fclose(pFile);

            //CopyFile(shm_name, const_cast<char *>(data_path.c_str()));
        }
    };
    REGISTER_KERNEL_BUILDER(Name("SaveNebula").Device(DEVICE_CPU), SaveNebula);
}