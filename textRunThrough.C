#include "/user/vlukic/software/RadioScatter/include/RadioScatter.hh"
#include "/user/vlukic/software/RadioScatter/include/TUtilRadioScatter.hh"
#include "radioScatterEventToText2.C"

R__ADD_LIBRARY_PATH("/user/vlukic/software/RadioScatter_install_dir/lib/") // if needed
R__LOAD_LIBRARY(libRadioScatter.so)

void textRunThrough(TString infile, int begin, int runs, int txindex, int rxindex){
   for(int j=begin;j<runs;j++){
       radioScatterEventToText2(infile,j,txindex,rxindex);
      }
}
