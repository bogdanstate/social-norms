/*
 * Copyright (C) 2015 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package com.google.cloud.dataflow.examples;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.cloud.dataflow.examples.XMLEditHistory.Page;
import com.google.cloud.dataflow.sdk.Pipeline;
import com.google.cloud.dataflow.sdk.io.BigQueryIO;
import com.google.cloud.dataflow.sdk.io.Read;
import com.google.cloud.dataflow.sdk.io.TextIO;
import com.google.cloud.dataflow.sdk.io.XmlSource;
import com.google.cloud.dataflow.sdk.io.CompressedSource;
import com.google.cloud.dataflow.sdk.io.FileBasedSource;
import com.google.cloud.dataflow.sdk.options.DataflowPipelineOptions;
import com.google.cloud.dataflow.sdk.options.Default;
import com.google.cloud.dataflow.sdk.options.DefaultValueFactory;
import com.google.cloud.dataflow.sdk.options.Description;
import com.google.cloud.dataflow.sdk.options.PipelineOptions;
import com.google.cloud.dataflow.sdk.options.PipelineOptionsFactory;
import com.google.cloud.dataflow.sdk.transforms.Aggregator;
import com.google.cloud.dataflow.sdk.transforms.Count;
import com.google.cloud.dataflow.sdk.transforms.DoFn;
import com.google.cloud.dataflow.sdk.transforms.PTransform;
import com.google.cloud.dataflow.sdk.transforms.ParDo;
import com.google.cloud.dataflow.sdk.transforms.Sum;
import com.google.cloud.dataflow.sdk.transforms.DoFn.ProcessContext;
import com.google.cloud.dataflow.sdk.util.gcsfs.GcsPath;
import com.google.cloud.dataflow.sdk.values.KV;
import com.google.cloud.dataflow.sdk.values.PCollection;

import java.util.ArrayList;
import java.util.List;

import javax.lang.model.element.Element;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAnyElement;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlTransient;
import javax.xml.bind.annotation.XmlType;


public class XMLEditHistory {

	@XmlRootElement(name = "contributor")
	@XmlAccessorType(XmlAccessType.FIELD)
	public static class Contributor {
		@XmlElement(name = "username")
		String username;
		@XmlElement(name = "id")
		Integer userid;
		
		public String getUsername() {
			return this.username != null ? this.username : "";
		}
		
		public Integer getUserid() {
			return this.userid != null ? this.userid : -1;
		}
		
		public void augmentRow(TableRow row) {
			row.set("username", this.username);
			row.set("userid", this.userid);
		}
		
		@XmlAnyElement(lax=true)
		public List<Object> others;
	
	}
	
	@XmlRootElement(name = "revision")
	@XmlAccessorType(XmlAccessType.FIELD)
	public static class Revision {

		@XmlElement(name = "id")
		Integer id;
		@XmlElement(name = "comment")
		String comment;
		@XmlElement(name = "timestamp")
		String timestamp;
		@XmlElement(name = "contributor")
		Contributor contributor;
		
		@XmlTransient
		private String text;

		@XmlTransient
		private String model;
		
		public void setComment(String comment) {
			this.comment = comment;
		}
		
		public void augmentRow(TableRow row) {
			row.set("revisionid", this.id);
			row.set("comment", this.comment);
			row.set("timestamp", this.timestamp);
			if (this.contributor != null) {
				this.contributor.augmentRow(row);
			}
		}
		
		//public List<Object> others;
		
		public void setOthers(List<Object> others) {
			//this.others = others;
		}
		@XmlAnyElement(lax=true)
		public List<Object> getOthers() {
			return null;
		}
	
	}
	
	@XmlRootElement(name = "page")
	@XmlAccessorType(XmlAccessType.NONE)
	public static class Page {
	  
		@XmlElement(name = "id")
		Integer id;
		@XmlElement(name = "title")
		String title;
	    		
		@XmlElement(name = "revision")
	    public List<Revision> revisions;
	    
	    
	    //	@XmlElement(name = "title")
	    public void setTitle(String title) {
	    	this.title = title;
	    }
	    
	    public String getTitle() {
	    	return this.title;
	    }
		
	    public void setRevisions(List<Revision> revisions) {
	    	this.revisions = revisions;
	    }
	    
	    public List<Revision> getRevisions() {
	    	return this.revisions;
	    }
	    
		@XmlAnyElement(lax=true)
		public List<Object> others;
	

  }

  static class ExtractTitleFn extends DoFn<Page, TableRow> {

    @Override
    public void processElement(ProcessContext c) {
      if (c.element().getRevisions() != null) {
    	  for (Revision rev : c.element().getRevisions()) {
    		  TableRow row = new TableRow();
    		  row.set("pageid", c.element().id);
    		  rev.augmentRow(row);
    		  c.output(row);
    	  }
      }
    }
  }

  public static interface XMLEditHistoryOptions extends PipelineOptions {
    @Description("Path of the file to read from")
    @Default.String("gs://dataflow-samples/shakespeare/kinglear.txt")
    String getInputFile();
    void setInputFile(String value);

    @Description("Path of the file to write to")
    @Default.InstanceFactory(OutputFactory.class)
    String getOutput();
    void setOutput(String value);

    /**
     * Returns "gs://${YOUR_STAGING_DIRECTORY}/counts.txt" as the default destination.
     */
    public static class OutputFactory implements DefaultValueFactory<String> {
      @Override
      public String create(PipelineOptions options) {
        DataflowPipelineOptions dataflowOptions = options.as(DataflowPipelineOptions.class);
        if (dataflowOptions.getStagingLocation() != null) {
          return GcsPath.fromUri(dataflowOptions.getStagingLocation())
              .resolve("counts.txt").toString();
        } else {
          throw new IllegalArgumentException("Must specify --output or --stagingLocation");
        }
      }
    }

  }

  public static void main(String[] args) throws Exception {

	 XMLEditHistoryOptions options = PipelineOptionsFactory.fromArgs(args).withValidation()
	    	      .as(XMLEditHistoryOptions.class);
	 Pipeline p = Pipeline.create(options);
	  
	 FileBasedSource<Page> source = CompressedSource.from(
			 XmlSource.<Page>from(options.getInputFile())
			 .withRootElement("mediawiki")
			 .withMinBundleSize(100)
			 .withRecordElement("page")
			 .withRecordClass(Page.class)
	).withDecompression(CompressedSource.CompressionMode.BZIP2);
	 

	 List<TableFieldSchema> fields = new ArrayList<>();
	 fields.add(new TableFieldSchema().setName("pageid").setType("INTEGER"));
	 fields.add(new TableFieldSchema().setName("userid").setType("INTEGER"));
	 fields.add(new TableFieldSchema().setName("revisionid").setType("INTEGER"));
	 
	 fields.add(new TableFieldSchema().setName("comment").setType("STRING"));
	 fields.add(new TableFieldSchema().setName("timestamp").setType("STRING"));
	 fields.add(new TableFieldSchema().setName("username").setType("STRING"));
	 
	 TableSchema schema = new TableSchema().setFields(fields);
	 p.apply(
			 Read.from(source)
			 .named("Read")		 
	)
	  .apply(ParDo.of(new ExtractTitleFn()).named("Transform"))
	  .apply(BigQueryIO.Write.named("Write")
			  				 .withSchema(schema)
			  				 .to(options.getOutput())
	);
	p.run();
  }
}
