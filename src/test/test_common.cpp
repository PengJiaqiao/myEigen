// g++ -std=c++11 test_common.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_common

#include "common.hpp"

char log_path[] = "log";

using namespace MRI;

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    GPU_Controller::SetDevice(0);
    GPU_Controller::DeviceQuery();
    CHECK_EQ(GPU_Controller::CheckDevice(0), true);
    CHECK_EQ(GPU_Controller::FindDevice(0), 0);
    GPU_Controller::cuFFT_handle();
    GPU_Controller::cuSPARSE_handle();
    printf("%s\n", "Testing passed.");

    return 0;
}