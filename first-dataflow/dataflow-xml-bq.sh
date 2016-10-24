 mvn compile exec:java \
      -Dexec.mainClass=com.google.cloud.dataflow.examples.XMLEditHistory \
      -Dexec.args="--project=social-norms \
      --output=social-norms:wikipedia.revisions_full \
      --stagingLocation=gs://social-norms/staging/ \
      --tempLocation=gs://social-norms/temp/ \
      --inputFile=gs://social-norms/wikidumps/enwiki-latest-pages-meta-history*.bz2 \
      --runner=BlockingDataflowPipelineRunner"
      #--inputFile=/home/bogdan/Downloads/Wikipedia-20161021025047.xml.gz"
      #--inputFile=wikipedia.tar.gz"
#--inputFile=test_input.txt" -X
      #--inputFile=canonical.txt"
#--output=gs://social-norms/output \
