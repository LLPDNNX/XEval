#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/core/graph/default_device.h"

#include <exception>

#include "TTree.h"
#include "TFile.h"

class NanoxTree
{
    public:
        long ientry_;
        TTree* tree_;
    
        static constexpr int maxEntries = 250; //25*10 -> allows for a maximum of 10 jets per event
        
        unsigned int nJet;
        float Jet_eta[maxEntries];
        float Jet_pt[maxEntries];
        unsigned int Jet_jetId[maxEntries];
        unsigned int Jet_cleanmask[maxEntries];
        
        unsigned int njetorigin;
        float jetorigin_isPU[maxEntries];
        float jetorigin_isUndefined[maxEntries];
        
        float jetorigin_displacement[maxEntries];
        float jetorigin_decay_angle[maxEntries];
        
        float jetorigin_isB[maxEntries];
        float jetorigin_isBB[maxEntries];
        float jetorigin_isGBB[maxEntries];
        float jetorigin_isLeptonic_B[maxEntries];
        float jetorigin_isLeptonic_C[maxEntries];
        float jetorigin_isC[maxEntries];
        float jetorigin_isCC[maxEntries];
        float jetorigin_isGCC[maxEntries];
        float jetorigin_isS[maxEntries];
        float jetorigin_isUD[maxEntries];
        float jetorigin_isG[maxEntries];
        float jetorigin_fromLLP[maxEntries];
        
        //float jetorigin_llpmass_reco[maxEntries];
        
        unsigned int nglobal;
        float global_pt[maxEntries];
        float global_eta[maxEntries];
        float global_rho;
        
        unsigned int ncpflength;
        float cpflength_length[maxEntries];
        
        unsigned int ncpf[maxEntries];
        float cpf_trackEtaRel[maxEntries];
        float cpf_trackPtRel[maxEntries];
        float cpf_trackPPar[maxEntries];
        float cpf_trackDeltaR[maxEntries];
        float cpf_trackPtRatio[maxEntries];
        float cpf_trackPParRatio[maxEntries];
        float cpf_trackSip2dVal[maxEntries];
        float cpf_trackSip2dSig[maxEntries];
        float cpf_trackSip3dVal[maxEntries];
        float cpf_trackSip3dSig[maxEntries];
        float cpf_trackJetDistVal[maxEntries];
        float cpf_trackJetDistSig[maxEntries];
        float cpf_ptrel[maxEntries];
        float cpf_drminsv[maxEntries];
        float cpf_vertex_association[maxEntries];
        float cpf_puppi_weight[maxEntries];
        float cpf_track_chi2[maxEntries];
        float cpf_track_quality[maxEntries];
        float cpf_jetmassdroprel[maxEntries];
        float cpf_relIso01[maxEntries];
        
        unsigned int ncsv[maxEntries];
        float csv_trackSumJetEtRatio[maxEntries];
        float csv_trackSumJetDeltaR[maxEntries];
        float csv_vertexCategory[maxEntries];
        float csv_trackSip2dValAboveCharm[maxEntries];
        float csv_trackSip2dSigAboveCharm[maxEntries];
        float csv_trackSip3dValAboveCharm[maxEntries];
        float csv_trackSip3dSigAboveCharm[maxEntries];
        float csv_jetNSelectedTracks[maxEntries];
        float csv_jetNTracksEtaRel[maxEntries];
        
        unsigned int nnpflength;
        float npflength_length[maxEntries];
        
        unsigned int nnpf[maxEntries];
        float npf_ptrel[maxEntries];
        float npf_deltaR[maxEntries];
        float npf_isGamma[maxEntries];
        float npf_hcal_fraction[maxEntries];
        float npf_drminsv[maxEntries];
        float npf_puppi_weight[maxEntries];
        float npf_jetmassdroprel[maxEntries];
        float npf_relIso01[maxEntries];
        
        unsigned int nsvlength;
        float svlength_length[maxEntries];
        
        unsigned int nsv[maxEntries];
        float sv_pt[maxEntries];
        float sv_mass[maxEntries];
        float sv_deltaR[maxEntries];
        float sv_ntracks[maxEntries];
        float sv_chi2[maxEntries];
        float sv_normchi2[maxEntries];
        float sv_dxy[maxEntries];
        float sv_dxysig[maxEntries];
        float sv_d3d[maxEntries];
        float sv_d3dsig[maxEntries];
        float sv_costhetasvpv[maxEntries];
        float sv_enratio[maxEntries];
        
        NanoxTree(TTree* tree):
            ientry_(0),
            tree_(tree)
        {
            tree_->SetBranchAddress("nJet",&nJet);
            tree_->SetBranchAddress("Jet_eta",&Jet_eta);
            tree_->SetBranchAddress("Jet_pt",&Jet_pt);
            tree_->SetBranchAddress("Jet_jetId",&Jet_jetId);
            tree_->SetBranchAddress("Jet_cleanmask",&Jet_cleanmask);
        
            tree_->SetBranchAddress("njetorigin",&njetorigin);
            
            tree_->SetBranchAddress("jetorigin_isPU",&jetorigin_isPU);
            tree_->SetBranchAddress("jetorigin_isUndefined",&jetorigin_isUndefined);
            
            tree_->SetBranchAddress("jetorigin_displacement",&jetorigin_displacement);
            tree_->SetBranchAddress("jetorigin_decay_angle",&jetorigin_decay_angle);
            
            tree_->SetBranchAddress("jetorigin_isB",&jetorigin_isB);
            tree_->SetBranchAddress("jetorigin_isBB",&jetorigin_isBB);
            tree_->SetBranchAddress("jetorigin_isGBB",&jetorigin_isGBB);
            tree_->SetBranchAddress("jetorigin_isLeptonic_B",&jetorigin_isLeptonic_B);
            tree_->SetBranchAddress("jetorigin_isLeptonic_C",&jetorigin_isLeptonic_C);
            tree_->SetBranchAddress("jetorigin_isC",&jetorigin_isC);
            tree_->SetBranchAddress("jetorigin_isCC",&jetorigin_isCC);
            tree_->SetBranchAddress("jetorigin_isGCC",&jetorigin_isGCC);
            tree_->SetBranchAddress("jetorigin_isS",&jetorigin_isS);
            tree_->SetBranchAddress("jetorigin_isUD",&jetorigin_isUD);
            tree_->SetBranchAddress("jetorigin_isG",&jetorigin_isG);
            tree_->SetBranchAddress("jetorigin_fromLLP",&jetorigin_fromLLP);
            
            //tree_->SetBranchAddress("jetorigin_llpmass_reco",&jetorigin_llpmass_reco);
            
            tree_->SetBranchAddress("nglobal",&nglobal);
            tree_->SetBranchAddress("global_pt",&global_pt);
            tree_->SetBranchAddress("global_eta",&global_eta);
            tree_->SetBranchAddress("fixedGridRhoFastjetAll",&global_rho);
            
            tree_->SetBranchAddress("ncpflength",&ncpflength);
            tree_->SetBranchAddress("cpflength_length",&cpflength_length);
            
            tree_->SetBranchAddress("ncpf",&ncpf);
            tree_->SetBranchAddress("cpf_trackEtaRel",&cpf_trackEtaRel);
            tree_->SetBranchAddress("cpf_trackPtRel",&cpf_trackPtRel);
            tree_->SetBranchAddress("cpf_trackPPar",&cpf_trackPPar);
            tree_->SetBranchAddress("cpf_trackDeltaR",&cpf_trackDeltaR);
            tree_->SetBranchAddress("cpf_trackPtRatio",&cpf_trackPtRatio);
            tree_->SetBranchAddress("cpf_trackPParRatio",&cpf_trackPParRatio);
            tree_->SetBranchAddress("cpf_trackSip2dVal",&cpf_trackSip2dVal);
            tree_->SetBranchAddress("cpf_trackSip2dSig",&cpf_trackSip2dSig);
            tree_->SetBranchAddress("cpf_trackSip3dVal",&cpf_trackSip3dVal);
            tree_->SetBranchAddress("cpf_trackSip3dSig",&cpf_trackSip3dSig);
            tree_->SetBranchAddress("cpf_trackJetDistVal",&cpf_trackJetDistVal);
            tree_->SetBranchAddress("cpf_trackJetDistSig",&cpf_trackJetDistSig);
            tree_->SetBranchAddress("cpf_ptrel",&cpf_ptrel);
            tree_->SetBranchAddress("cpf_drminsv",&cpf_drminsv);
            tree_->SetBranchAddress("cpf_vertex_association",&cpf_vertex_association);
            tree_->SetBranchAddress("cpf_puppi_weight",&cpf_puppi_weight);
            tree_->SetBranchAddress("cpf_track_chi2",&cpf_track_chi2);
            tree_->SetBranchAddress("cpf_track_quality",&cpf_track_quality);
            tree_->SetBranchAddress("cpf_jetmassdroprel",&cpf_jetmassdroprel);
            tree_->SetBranchAddress("cpf_relIso01",&cpf_relIso01);
            
            tree_->SetBranchAddress("ncsv",&ncsv);
            tree_->SetBranchAddress("csv_trackSumJetEtRatio",&csv_trackSumJetEtRatio);
            tree_->SetBranchAddress("csv_trackSumJetDeltaR",&csv_trackSumJetDeltaR);
            tree_->SetBranchAddress("csv_vertexCategory",&csv_vertexCategory);
            tree_->SetBranchAddress("csv_trackSip2dValAboveCharm",&csv_trackSip2dValAboveCharm);
            tree_->SetBranchAddress("csv_trackSip2dSigAboveCharm",&csv_trackSip2dSigAboveCharm);
            tree_->SetBranchAddress("csv_trackSip3dValAboveCharm",&csv_trackSip3dValAboveCharm);
            tree_->SetBranchAddress("csv_trackSip3dSigAboveCharm",&csv_trackSip3dSigAboveCharm);
            tree_->SetBranchAddress("csv_jetNSelectedTracks",&csv_jetNSelectedTracks);
            tree_->SetBranchAddress("csv_jetNTracksEtaRel",&csv_jetNTracksEtaRel);
           
            tree_->SetBranchAddress("nnpflength",&nnpflength);
            tree_->SetBranchAddress("npflength_length",&npflength_length);
            
            tree_->SetBranchAddress("nnpf",&nnpf);
            tree_->SetBranchAddress("npf_ptrel",&npf_ptrel);
            tree_->SetBranchAddress("npf_deltaR",&npf_deltaR);
            tree_->SetBranchAddress("npf_isGamma",&npf_isGamma);
            tree_->SetBranchAddress("npf_hcal_fraction",&npf_hcal_fraction);
            tree_->SetBranchAddress("npf_drminsv",&npf_drminsv);
            tree_->SetBranchAddress("npf_puppi_weight",&npf_puppi_weight);
            tree_->SetBranchAddress("npf_jetmassdroprel",&npf_jetmassdroprel);
            tree_->SetBranchAddress("npf_relIso01",&npf_relIso01);
            
            tree_->SetBranchAddress("nsvlength",&nsvlength);
            tree_->SetBranchAddress("svlength_length",&svlength_length);
            
            tree_->SetBranchAddress("nsv",&nsv);
            tree_->SetBranchAddress("sv_pt",&sv_pt);
            tree_->SetBranchAddress("sv_mass",&sv_mass);
            tree_->SetBranchAddress("sv_deltaR",&sv_deltaR);
            tree_->SetBranchAddress("sv_ntracks",&sv_ntracks);
            tree_->SetBranchAddress("sv_chi2",&sv_chi2);
            tree_->SetBranchAddress("sv_normchi2",&sv_normchi2);
            tree_->SetBranchAddress("sv_dxy",&sv_dxy);
            tree_->SetBranchAddress("sv_dxysig",&sv_dxysig);
            tree_->SetBranchAddress("sv_d3d",&sv_d3d);
            tree_->SetBranchAddress("sv_d3dsig",&sv_d3dsig);
            tree_->SetBranchAddress("sv_costhetasvpv",&sv_costhetasvpv);
            tree_->SetBranchAddress("sv_enratio",&sv_enratio);
            
            tree_->GetEntry(0);
        }
        
        inline unsigned int entries() const
        {
            return tree_->GetEntries();
        }
        
        inline unsigned int entry() const
        {
            return ientry_;
        }
        
        bool getEvent(unsigned int entry, bool force=false)
        {
            if (force or entry!=ientry_)
            {
                tree_->GetEntry(entry);
                ientry_ = entry;
                return true;
            }
            if (entry>=entries())
            {
                return false;
            }
            return true;
        }
        
        bool nextEvent()
        {
            return getEvent(ientry_+1);
        }
       
        inline int njets()
        {
            return nglobal; //nanox training data only stored for jets pT>20 GeV (= subset of nJet)
        }
        
        inline static float resetNanOfInf(float x)
        {
            if (std::isnan(x) or std::isinf(x)) return 0;
            return x;
        }
        
        int getJetClass(unsigned int entry,unsigned int jet)
        {
            getEvent(entry);
            //if (jetorigin_isPU[jet]>0.5) return 11;
            if (jetorigin_fromLLP[jet]<0.5)
            {
                if  (jetorigin_isB[jet]>0.5) return 0;
                if  (jetorigin_isBB[jet]>0.5) return 0;
                if  (jetorigin_isGBB[jet]>0.5) return 0;
                if  (jetorigin_isLeptonic_B[jet]>0.5) return 0;
                if  (jetorigin_isLeptonic_C[jet]>0.5) return 0;
                if  (jetorigin_isC[jet]>0.5) return 1;
                if  (jetorigin_isCC[jet]>0.5) return 1;
                if  (jetorigin_isGCC[jet]>0.5) return 1;
                if  (jetorigin_isS[jet]>0.5) return 2;
                if  (jetorigin_isUD[jet]>0.5) return 2;
                if  (jetorigin_isG[jet]>0.5) return 3;
            }
            else
            {
                return 4;
            }
            return -1;
        }
        
        bool isSelected(unsigned int entry,unsigned int jet)
        {
            getEvent(entry);
            if (Jet_pt[jet]<20.) return false;
            if (std::fabs(Jet_eta[jet])>2.4) return false;
            return true;
        }
        
        void fillTensors(
            tensorflow::Tensor& cpf_tensor,
            tensorflow::Tensor& npf_tensor,
            tensorflow::Tensor& sv_tensor,
            tensorflow::Tensor& globalvars_tensor,
            unsigned int entry,
            unsigned int jet
        )
        {
            getEvent(entry);

            
            auto globalvars = globalvars_tensor.tensor<float,2>();
           
            //std::cout<<entry<<"/"<<jet<<": "<< global_pt[jet]<<"/"<<global_eta[jet]<<std::endl;
            globalvars(0,0) = global_pt[jet];
            globalvars(0,1) = global_eta[jet];
            globalvars(0,2) = global_rho;
            
            globalvars(0,3) = cpflength_length[jet];
            globalvars(0,4) = npflength_length[jet];
            globalvars(0,5) = svlength_length[jet];
            
            globalvars(0,6) = csv_trackSumJetEtRatio[jet];
            
            globalvars(0,7) = csv_trackSumJetDeltaR[jet];
            globalvars(0,8) = csv_vertexCategory[jet];
            globalvars(0,9) = csv_trackSip2dValAboveCharm[jet];
            globalvars(0,10) = csv_trackSip2dSigAboveCharm[jet];
            globalvars(0,11) = csv_trackSip3dValAboveCharm[jet];
            globalvars(0,12) = csv_trackSip3dSigAboveCharm[jet];
            globalvars(0,13) = csv_jetNSelectedTracks[jet];
            globalvars(0,14) = csv_jetNTracksEtaRel[jet];
            
            
            
            auto cpf = cpf_tensor.tensor<float,3>();
            
            int cpf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                cpf_offset += cpflength_length[i];
            }
            
            int ncpf = std::min<int>(cpf_tensor.dim_size(1),cpflength_length[jet]);
            for (int i = 0; i < ncpf; ++i)
            {
                
                cpf(0,i,0) = cpf_trackEtaRel[cpf_offset+i];
                cpf(0,i,1) = cpf_trackPtRel[cpf_offset+i];
                cpf(0,i,2) = cpf_trackPPar[cpf_offset+i];
                cpf(0,i,3) = cpf_trackDeltaR[cpf_offset+i];
                cpf(0,i,4) = cpf_trackPParRatio[cpf_offset+i];
                
                //unpackedTree.cpf_trackPtRatio[i] = cpf_trackPtRatio[cpf_offset+i];
                
                cpf(0,i,5) = cpf_trackSip2dVal[cpf_offset+i];
                cpf(0,i,6) = resetNanOfInf(cpf_trackSip2dSig[cpf_offset+i]);
                cpf(0,i,7) = cpf_trackSip3dVal[cpf_offset+i];
                cpf(0,i,8) = resetNanOfInf(cpf_trackSip3dSig[cpf_offset+i]);
                cpf(0,i,9) = cpf_trackJetDistVal[cpf_offset+i];
                
                //unpackedTree.cpf_trackJetDistSig[i] = cpf_trackJetDistSig[cpf_offset+i];
                cpf(0,i,10) = cpf_ptrel[cpf_offset+i];
                cpf(0,i,11) = cpf_drminsv[cpf_offset+i];
                cpf(0,i,12) = cpf_vertex_association[cpf_offset+i];
                cpf(0,i,13) = cpf_puppi_weight[cpf_offset+i];
                cpf(0,i,14) = cpf_track_chi2[cpf_offset+i];
                cpf(0,i,15) = cpf_track_quality[cpf_offset+i];
                //unpackedTree.cpf_jetmassdroprel[i] = cpf_jetmassdroprel[cpf_offset+i];
                //unpackedTree.cpf_relIso01[i] = cpf_relIso01[cpf_offset+i];
                
            }
            for (int i = ncpf; i < cpf_tensor.dim_size(1); ++i)
            {
                cpf(0,i,0) = 0;
                cpf(0,i,1) = 0;
                cpf(0,i,2) = 0;
                cpf(0,i,3) = 0;
                cpf(0,i,4) = 0;
                //unpackedTree.cpf_trackPtRatio[i] = cpf_trackPtRatio[cpf_offset+i];
                cpf(0,i,5) = 0;
                cpf(0,i,6) = 0;
                cpf(0,i,7) = 0;
                cpf(0,i,8) = 0;
                cpf(0,i,9) = 0;
                //unpackedTree.cpf_trackJetDistSig[i] = cpf_trackJetDistSig[cpf_offset+i];
                cpf(0,i,10) = 0;
                cpf(0,i,11) = 0;
                cpf(0,i,12) = 0;
                cpf(0,i,13) = 0;
                cpf(0,i,14) = 0;
                cpf(0,i,15) = 0;
                //unpackedTree.cpf_jetmassdroprel[i] = cpf_jetmassdroprel[cpf_offset+i];
                //unpackedTree.cpf_relIso01[i] = cpf_relIso01[cpf_offset+i];
            }
            
            
            auto npf = npf_tensor.tensor<float,3>();
            int npf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                npf_offset += npflength_length[i];
            }
            
            int nnpf = std::min<int>(npf_tensor.dim_size(1),npflength_length[jet]);
            for (int i = 0; i < nnpf; ++i)
            {
                
                npf(0,i,0) = npf_ptrel[npf_offset+i];
                npf(0,i,1) = npf_deltaR[npf_offset+i];
                npf(0,i,2) = npf_isGamma[npf_offset+i];
                npf(0,i,3) = npf_hcal_fraction[npf_offset+i];
                npf(0,i,4) = npf_drminsv[npf_offset+i];
                npf(0,i,5) = npf_puppi_weight[npf_offset+i];
                
                //unpackedTree.npf_jetmassdroprel[i] = npf_jetmassdroprel[npf_offset+i];
                //unpackedTree.npf_relIso01[i] = npf_relIso01[npf_offset+i];
            }
            for (int i = nnpf; i < npf_tensor.dim_size(1); ++i)
            {
                npf(0,i,0) = 0;
                npf(0,i,1) = 0;
                npf(0,i,2) = 0;
                npf(0,i,3) = 0;
                npf(0,i,4) = 0;
                npf(0,i,5) = 0;
                //unpackedTree.npf_jetmassdroprel[i] = npf_jetmassdroprel[npf_offset+i];
                //unpackedTree.npf_relIso01[i] = npf_relIso01[npf_offset+i];
            }
            
            
            auto sv = sv_tensor.tensor<float,3>();
            
            int sv_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                sv_offset += svlength_length[i];
            }
            
            int nsv = std::min<int>(sv_tensor.dim_size(1),svlength_length[jet]);
            for (int i = 0; i < nsv; ++i)
            {
                
                sv(0,i,0) = sv_pt[sv_offset+i];
                sv(0,i,1) = sv_deltaR[sv_offset+i];
                sv(0,i,2) = sv_mass[sv_offset+i];
                sv(0,i,3) = sv_ntracks[sv_offset+i];
                sv(0,i,4) = sv_chi2[sv_offset+i];
                
                sv(0,i,5) = resetNanOfInf(sv_normchi2[sv_offset+i]);
                
                sv(0,i,6) = sv_dxy[sv_offset+i];
                sv(0,i,7) = sv_dxysig[sv_offset+i];
                sv(0,i,8) = sv_d3d[sv_offset+i];
                sv(0,i,9) = sv_d3dsig[sv_offset+i];
                
                sv(0,i,10) = sv_costhetasvpv[sv_offset+i];
                sv(0,i,11) = sv_enratio[sv_offset+i];
                
            }
            
            for (int i = nsv; i < sv_tensor.dim_size(1); ++i)
            {
                sv(0,i,0) = 0;
                sv(0,i,1) = 0;
                sv(0,i,2) = 0;
                sv(0,i,3) = 0;
                sv(0,i,4) = 0;
                sv(0,i,5) = 0;
                sv(0,i,6) = 0;
                sv(0,i,7) = 0;
                sv(0,i,8) = 0;
                sv(0,i,9) = 0;
                sv(0,i,10) = 0;
                sv(0,i,11) = 0;
            }
            
            
        }
};

int main(int argc, char *argv[])
{
    std::cout<<"Input file: "<<argv[1]<<std::endl;
    std::cout<<"Output file: "<<argv[2]<<std::endl;
    
    tensorflow::Status status;

    // load it
    tensorflow::GraphDef graphDef;
    status = ReadBinaryProto(tensorflow::Env::Default(), "nanox_ctau_1_new.pb", &graphDef);
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
    
    tensorflow::Tensor cpf(tensorflow::DT_FLOAT, {1,25,16});
    tensorflow::Tensor npf(tensorflow::DT_FLOAT, {1,25,6});
    tensorflow::Tensor sv(tensorflow::DT_FLOAT, {1,4,12});
    tensorflow::Tensor globalvars(tensorflow::DT_FLOAT, {1,15});
    
    TFile outputFile(argv[2],"RECREATE");
    TTree outputTree("Events","Events");
    outputTree.SetDirectory(&outputFile);
    outputTree.SetAutoSave(10000);
    
    int nllpdnnx = 0;
    float isB[50];
    float isC[50];
    float isUDS[50];
    float isG[50];
    float isLLP[50];
    outputTree.Branch("nllpdnnx",&nllpdnnx,"nllpdnnx/I",64000);
    outputTree.Branch("llpdnnx_isB",&isB,"llpdnnx_isB[nllpdnnx]/F",64000);
    outputTree.Branch("llpdnnx_isC",&isC,"llpdnnx_isC[nllpdnnx]/F",64000);
    outputTree.Branch("llpdnnx_isUDS",&isUDS,"llpdnnx_isUDS[nllpdnnx]/F",64000);
    outputTree.Branch("llpdnnx_isG",&isG,"llpdnnx_isG[nllpdnnx]/F",64000);
    outputTree.Branch("llpdnnx_isLLP",&isLLP,"llpdnnx_isLLP[nllpdnnx]/F",64000);
    
    std::unique_ptr<TFile> file(TFile::Open(
        //"root://gfe02.grid.hep.ph.ic.ac.uk/pnfs/hep.ph.ic.ac.uk/data/cms/store/user/mkomm/LLP/NANOX_180425-v2/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8-evtgen/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8-evtgen/NANOX_180425-v2/180425_183639/0000/nano_20.root"
        argv[1]
    ));
    TTree* tree = (TTree*)file->Get("Events");
    
    NanoxTree nanoxTree(tree);
    
    std::array<float,5> correct;
    std::array<float,5> total;
    for (int event = 0; event < tree->GetEntries(); ++event)
    {
        if (event%2000==0)
        {
            std::cout<<"processing "<<(100.*event/tree->GetEntries())<<"% ..."<<std::endl;
        }
        nanoxTree.getEvent(event);
        nllpdnnx = nanoxTree.njets();
        for (int jet = 0; jet < nanoxTree.njets(); ++jet)
        {
            if (not nanoxTree.isSelected(event,jet))
            {
                isB[jet] = -1;
                isC[jet] = -1;
                isUDS[jet] = -1;
                isG[jet] = -1;
                isLLP[jet] = -1;
                continue;
            }
            nanoxTree.fillTensors(
                cpf,npf,sv,globalvars,event,jet
            );
            
            std::vector<tensorflow::Tensor> outputs; 
            TF_CHECK_OK(session->Run(
                {
                    {"cpf",cpf},
                    {"npf",npf},
                    {"sv",sv},
                    {"globalvars",globalvars}
                }, //input map
                {"prediction"}, //output node names 
                {}, //additional nodes run but not put in outputs
                &outputs
            ));
            
            
            auto tensor_flat = outputs[0].flat<float>();
            float sum = 0.;
            int trueClass = nanoxTree.getJetClass(event,jet);
            int predictedClass = -1;
            float maxProb = -1;
            
            isB[jet] = tensor_flat(0);
            isC[jet] = tensor_flat(1);
            isUDS[jet] = tensor_flat(2);
            isG[jet] = tensor_flat(3);
            isLLP[jet] = tensor_flat(4);
            
            for (int i = 0; i < tensor_flat.size(); ++i)
            {
                if (maxProb<tensor_flat(i))
                {
                    maxProb = tensor_flat(i);
                    predictedClass = i;
                }
                //std::cout<<tensor_flat(i)<<", ";
                sum+=tensor_flat(i);
            }
            //std::cout<<" => "<<sum<<std::endl;
            
            if (trueClass>=0)
            {
                if (trueClass==predictedClass)
                {
                    correct[trueClass]+=1;
                }
                total[trueClass]+=1;
                
            }
            
        }
        outputTree.Fill();
    }
    
    std::cout<<"Total events processed: "<<tree->GetEntries()<<std::endl;
    std::cout<<"Total events written: "<<outputTree.GetEntries()<<std::endl;
    
    outputFile.cd();
    outputTree.Write();
    outputFile.Close();
    
    for (int i = 0; i < 5; ++i)
    {
        std::cout<<"class: "<<i<<", correct="<<correct[i]<<", total="<<total[i]<<", fraction="<<correct[i]/total[i]<<std::endl;
    }
    
    return 0;
}

