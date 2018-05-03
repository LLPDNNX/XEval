#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/core/graph/default_device.h"

#include <exception>

#include "TTree.h"
#include "TFile.h"


int main()
{
    tensorflow::Status status;

    // load it
    tensorflow::GraphDef graphDef;
    status = ReadBinaryProto(tensorflow::Env::Default(), "testGraph.pb", &graphDef);
    tensorflow::graph::SetDefaultDevice("/cpu:0", &graphDef);
    
    // check for success
    if (!status.ok())
    {
        throw std::runtime_error("InvalidGraphDef: error while loading graph def: "+status.ToString());
    }
    
    tensorflow::Session* session;
    tensorflow::SessionOptions opts;
    opts.config.set_intra_op_parallelism_threads(1);
    opts.config.set_inter_op_parallelism_threads(1);
    TF_CHECK_OK(tensorflow::NewSession(opts, &session));
    TF_CHECK_OK(session->Create(graphDef));
    
    tensorflow::Tensor inx(tensorflow::DT_FLOAT, {1});
    inx.flat<float>()(0)=3;
    
    std::vector<tensorflow::Tensor> outputs; 
    TF_CHECK_OK(session->Run(
        {
            {"inx",inx}
        }, //input map
        {"out"}, //output node names 
        {}, //additional nodes run but not put in outputs
        &outputs
    ));
    for (const auto& tensor: outputs)
    {
        auto tensor_flat = tensor.flat<float>();
        for (int i = 0; i < tensor_flat.size(); ++i)
        {
            std::cout<<i<<": "<<tensor_flat(0)<<std::endl;
        }
    }
    
    return 0;
}

