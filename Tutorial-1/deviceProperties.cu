#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// Returns CUDA cores per SM based on compute capability.
int getCoresPerSM(int major, int minor)
{
    switch (major)
    {
    case 2: // Fermi
        return (minor == 1) ? 48 : 32;

    case 3: // Kepler
        return 192;

    case 5: // Maxwell
        return 128;

    case 6: // Pascal
        if (minor == 0)
            return 64;
        else
            return 128;

    case 7: // Volta / Turing
        return 64;

    case 8: // Ampere / Ada
        if (minor == 0)
            return 64;
        else
            return 128;

    case 9: // Hopper / newer architectures
        return 128;

    default:
        return 128;
    }
}

void printDeviceProperties(int device)
{
    cudaDeviceProp prop;

    cudaError_t error = cudaGetDeviceProperties(&prop, device);

    if (error != cudaSuccess)
    {
        cerr << "Error getting device properties: "
             << cudaGetErrorString(error) << endl;
        return;
    }

    int coresPerSM = getCoresPerSM(prop.major, prop.minor);
    int totalCores = prop.multiProcessorCount * coresPerSM;

    cout << "CUDA DEVICE INFORMATION" << endl;

    cout << "Basic Information";

    cout << "Device Number: " << device << endl;
    cout << "Device Name:   " << prop.name << endl;

    cout << "Compute Capability:            "
         << prop.major << "." << prop.minor << endl;

    cout << "CUDA Cores:    " << totalCores << endl;

    cout << "Streaming Multiprocessors: "
         << prop.multiProcessorCount << endl;

    cout << "CUDA Cores per SM: "
         << coresPerSM << endl;

    // Memory information
    cout << endl
         << "[Memory Information]" << endl;
    ;

    cout << "Total Global Memory:   "
         << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
         << " GB" << endl;

    cout << "Shared Memory per Block:   "
         << prop.sharedMemPerBlock / 1024
         << " KB" << endl;

    cout << "Shared Memory per SM:  "
         << prop.sharedMemPerMultiprocessor / 1024
         << " KB" << endl;

    cout << "Registers per Block:   "
         << prop.regsPerBlock << endl;

    cout << "Registers per SM:  "
         << prop.regsPerMultiprocessor << endl;

    cout << "L2 Cache Size: "
         << prop.l2CacheSize / 1024.0
         << " KB" << endl;

    cout << "Memory Bus Width:  "
         << prop.memoryBusWidth << " bits" << endl;

    cout << "Memory Clock Rate: "
         << prop.memoryClockRate / 1000.0
         << " MHz" << endl;

    // Thread information
    cout << endl
         << "[Thread Information]" << endl;
    ;

    cout << "Warp Size: "
         << prop.warpSize << endl;

    cout << "Max Threads per Block: "
         << prop.maxThreadsPerBlock << endl;

    cout << "Max Threads per SM:    "
         << prop.maxThreadsPerMultiProcessor << endl;

    cout << "Max Threads Dimension: "
         << prop.maxThreadsDim[0] << " x "
         << prop.maxThreadsDim[1] << " x "
         << prop.maxThreadsDim[2] << endl;

    cout << "Max Grid Dimension:    "
         << prop.maxGridSize[0] << " x "
         << prop.maxGridSize[1] << " x "
         << prop.maxGridSize[2] << endl;

    // Clock information
    cout << endl
         << "[Clock Information]" << endl;

    cout << "GPU Clock Rate:    "
         << prop.clockRate / 1000.0
         << " MHz" << endl;

    // Texture information
    cout << endl
         << "[Texture Information]" << endl;

    cout << "Texture Alignment: "
         << prop.textureAlignment << " bytes" << endl;

    cout << "Maximum Texture 1D Width:  "
         << prop.maxTexture1D << endl;

    cout << "Maximum Texture 2D Width:  "
         << prop.maxTexture2D[0] << endl;

    cout << "Maximum Texture 2D Height: "
         << prop.maxTexture2D[1] << endl;

    cout << "Maximum Texture 3D Width:  "
         << prop.maxTexture3D[0] << endl;

    cout << "Maximum Texture 3D Height: "
         << prop.maxTexture3D[1] << endl;

    cout << "Maximum Texture 3D Depth:  "
         << prop.maxTexture3D[2] << endl;

    // Hardware capabilities
    cout << endl
         << "[Hardware Capabilities]" << endl;

    cout << "Concurrent Kernels:    "
         << (prop.concurrentKernels ? "Yes" : "No") << endl;

    cout << "Async Engine Count:    "
         << prop.asyncEngineCount << endl;

    cout << "Unified Addressing:    "
         << (prop.unifiedAddressing ? "Yes" : "No") << endl;

    cout << "Managed Memory:    "
         << (prop.managedMemory ? "Yes" : "No") << endl;

    cout << "ECC Enabled:   "
         << (prop.ECCEnabled ? "Yes" : "No") << endl;

    cout << "Can Map Host Memory:   "
         << (prop.canMapHostMemory ? "Yes" : "No") << endl;

    cout << "Cooperative Launch:    "
         << (prop.cooperativeLaunch ? "Yes" : "No") << endl;

    cout << endl
         << "[Device Limits]" << endl;
    ;

    cout << "Max Blocks per SM: "
         << prop.maxBlocksPerMultiProcessor << endl;

    cout << "Maximum Resident Threads:  "
         << prop.maxThreadsPerMultiProcessor << endl;

    cout << "Maximum Shared Memory Block:   "
         << prop.sharedMemPerBlock / 1024
         << " KB" << endl;

    cout << "\n============================================================"
         << endl;
}

int main()
{
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        cerr << "CUDA Error: "
             << cudaGetErrorString(error)
             << endl;

        return 1;
    }

    cout << "Number of CUDA Devices Found: "
         << deviceCount << endl;

    int runtimeVersion;

    cudaRuntimeGetVersion(&runtimeVersion);

    cout << "CUDA Runtime Version: "
         << runtimeVersion / 1000
         << "."
         << (runtimeVersion % 1000) / 10
         << endl;

    cout << endl;

    if (deviceCount == 0)
    {
        cout << "No CUDA-enabled GPU found." << endl;
        return 0;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        printDeviceProperties(i);
    }

    return 0;
}