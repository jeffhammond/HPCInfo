#!/bin/bash

unset no_proxy

# save e.g. https://registrationcenter.intel.com/en/products/download/3470/ as xxx.html and run this on it:
#grep -wo '\<http.*parallel_studio.*\.tgz\"' xxx.html  | grep -v online | sed "s/\"//g"

mkdir -p $HOME/Downloads
cd $HOME/Downloads
BASE=http://registrationcenter-download.intel.com/akdlm/irc_nas/tec
for x in \
         16225/parallel_studio_xe_2020_cluster_edition.tgz         \
         15809/parallel_studio_xe_2019_update5_cluster_edition.tgz \
         15533/parallel_studio_xe_2019_update4_cluster_edition.tgz \
         15268/parallel_studio_xe_2019_update3_cluster_edition.tgz \
         15088/parallel_studio_xe_2019_update2_cluster_edition.tgz \
         14850/parallel_studio_xe_2019_update1_cluster_edition.tgz \
         13589/parallel_studio_xe_2019_cluster_edition.tgz         \
         13717/parallel_studio_xe_2018_update4_cluster_edition.tgz \
         12998/parallel_studio_xe_2018_update3_cluster_edition.tgz \
         12717/parallel_studio_xe_2018_update2_cluster_edition.tgz \
         12374/parallel_studio_xe_2018_update1_cluster_edition.tgz \
         12058/parallel_studio_xe_2018_cluster_edition.tgz         \
         13709/parallel_studio_xe_2017_update8.tgz                 \
         12856/parallel_studio_xe_2017_update7.tgz                 \
         12534/parallel_studio_xe_2017_update6.tgz                 \
         12138/parallel_studio_xe_2017_update5.tgz                 \
         11537/parallel_studio_xe_2017_update4.tgz                 \
         11460/parallel_studio_xe_2017_update3.tgz                 \
         11298/parallel_studio_xe_2017_update2.tgz                 \
         10973/parallel_studio_xe_2017_update1.tgz                 \
         9651/parallel_studio_xe_2017.tgz                          \
         9781/parallel_studio_xe_2016_update4.tgz                  \
         9061/parallel_studio_xe_2016_update3.tgz                  \
         8676/parallel_studio_xe_2016_update2.tgz                  \
         8365/parallel_studio_xe_2016_update1.tgz                  \
         7997/parallel_studio_xe_2016.tgz                          \
         8469/parallel_studio_xe_2015_update6.tgz                  \
         8127/parallel_studio_xe_2015_update5.tgz                  \
         7538/parallel_studio_xe_2015_update3.tgz                  \
         5207/parallel_studio_xe_2015_update2.tgz                  \
         4992/parallel_studio_xe_2015_update1.tgz                  \
         4584/parallel_studio_xe_2015.tgz                          \
    ; do
    wget ${BASE}/$x
done

for x in \
         parallel_studio_xe_2020_cluster_edition.tgz         \
         parallel_studio_xe_2019_update5_cluster_edition.tgz \
         parallel_studio_xe_2019_update4_cluster_edition.tgz \
         parallel_studio_xe_2019_update3_cluster_edition.tgz \
         parallel_studio_xe_2019_update2_cluster_edition.tgz \
         parallel_studio_xe_2019_update1_cluster_edition.tgz \
         parallel_studio_xe_2019_cluster_edition.tgz         \
         parallel_studio_xe_2018_update4_cluster_edition.tgz \
         parallel_studio_xe_2018_update3_cluster_edition.tgz \
         parallel_studio_xe_2018_update2_cluster_edition.tgz \
         parallel_studio_xe_2018_update1_cluster_edition.tgz \
         parallel_studio_xe_2018_cluster_edition.tgz         \
         parallel_studio_xe_2017_update8.tgz                 \
         parallel_studio_xe_2017_update7.tgz                 \
         parallel_studio_xe_2017_update6.tgz                 \
         parallel_studio_xe_2017_update5.tgz                 \
         parallel_studio_xe_2017_update4.tgz                 \
         parallel_studio_xe_2017_update3.tgz                 \
         parallel_studio_xe_2017_update2.tgz                 \
         parallel_studio_xe_2017_update1.tgz                 \
         parallel_studio_xe_2017.tgz                         \
         parallel_studio_xe_2016_update4.tgz                 \
         parallel_studio_xe_2016_update3.tgz                 \
         parallel_studio_xe_2016_update2.tgz                 \
         parallel_studio_xe_2016_update1.tgz                 \
         parallel_studio_xe_2016.tgz                         \
         parallel_studio_xe_2015_update6.tgz                 \
         parallel_studio_xe_2015_update5.tgz                 \
         parallel_studio_xe_2015_update3.tgz                 \
         parallel_studio_xe_2015_update2.tgz                 \
         parallel_studio_xe_2015_update1.tgz                 \
         parallel_studio_xe_2015.tgz                         \
     ; do
    tar -xzf $x
done

for x in \
         parallel_studio_xe_2020_cluster_edition         \
         parallel_studio_xe_2019_update5_cluster_edition \
         parallel_studio_xe_2019_update4_cluster_edition \
         parallel_studio_xe_2019_update3_cluster_edition \
         parallel_studio_xe_2019_update2_cluster_edition \
         parallel_studio_xe_2019_update1_cluster_edition \
         parallel_studio_xe_2019_cluster_edition         \
         parallel_studio_xe_2018_update4_cluster_edition \
         parallel_studio_xe_2018_update3_cluster_edition \
         parallel_studio_xe_2018_update2_cluster_edition \
         parallel_studio_xe_2018_update1_cluster_edition \
         parallel_studio_xe_2018_cluster_edition         \
         parallel_studio_xe_2017_update8                 \
         parallel_studio_xe_2017_update7                 \
         parallel_studio_xe_2017_update6                 \
         parallel_studio_xe_2017_update5                 \
         parallel_studio_xe_2017_update4                 \
         parallel_studio_xe_2017_update3                 \
         parallel_studio_xe_2017_update2                 \
         parallel_studio_xe_2017_update1                 \
         parallel_studio_xe_2017                         \
         parallel_studio_xe_2016_update4                 \
         parallel_studio_xe_2016_update3                 \
         parallel_studio_xe_2016_update2                 \
         parallel_studio_xe_2016_update1                 \
         parallel_studio_xe_2016                         \
         parallel_studio_xe_2015_update6                 \
         parallel_studio_xe_2015_update5                 \
         parallel_studio_xe_2015_update3                 \
         parallel_studio_xe_2015_update2                 \
         parallel_studio_xe_2015_update1                 \
         parallel_studio_xe_2015                         \
    ; do
    # prepare for silent install
    #sed -i "s/\/opt\/intel/\/\$HOME\/INTEL/g" $x/silent.cfg
    sed -i "s/decline/accept/g" $x/silent.cfg
    pushd $x
    ./install.sh --silent silent.cfg
    popd
done

