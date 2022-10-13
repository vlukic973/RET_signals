#include "/user/vlukic/software/RadioScatter/include/RadioScatter.hh"
#include "/user/vlukic/software/RadioScatter/include/TUtilRadioScatter.hh"

R__ADD_LIBRARY_PATH("/user/vlukic/software/RadioScatter_install_dir/lib/") // if needed
R__LOAD_LIBRARY(libRadioScatter.so)

void radioScatterEventToText2(TString infile, int entry, int txindex, int rxindex){
  auto rs=new RadioScatterEvent();
  auto ff=TFile::Open(infile);
  auto tree=(TTree*)ff->Get("tree");
  tree->SetBranchAddress("event", &rs);
  tree->GetEntry(entry);
  auto graph=rs->getGraph(txindex, rxindex);
  for(int i=0;i<graph->GetN();i++){
     printf("%e",graph->GetY()[i]);
     printf("\n"); 	
  }
}
