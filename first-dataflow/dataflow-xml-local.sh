 mvn compile exec:java \
      -Dexec.mainClass=com.google.cloud.dataflow.examples.XMLEditHistory \
      -Dexec.args="--project=social-norms \
      --output=my_output.txt \
      --inputFile=/home/bogdan/Downloads/Wikipedia-20161021025047.xml"
#--inputFile=example.xml"
#--inputFile=test_input.txt" -X
      #--inputFile=canonical.txt"
#--output=gs://social-norms/output \
#--stagingLocation=gs://social-norms/staging/ \
#--runner=BlockingDataflowPipelineRunner"
