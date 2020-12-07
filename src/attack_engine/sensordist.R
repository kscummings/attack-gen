library(tidyverse)
library(readr)

DATA_PATH<-"/Users/kaylacummings/Dropbox (MIT)/batadal"
TRIAL_DIR<-"interdiction_results"

res<-read_csv(file.path(DATA_PATH,TRIAL_DIR,"results.csv"))
res %>%
  group_by(budget,fortify,trial_type,trial_param) %>%
  mutate(s1=as.numeric(s1),
         s2=as.numeric(s2),
         s3=as.numeric(s3),
         s4=as.numeric(s4),
         s5=as.numeric(s5)) %>%
  summarise(s1=sum(s1,na.rm=TRUE),
            s2=sum(s2,na.rm=TRUE),
            s3=sum(s3,na.rm=TRUE),
            s4=sum(s4,na.rm=TRUE),
            s5=sum(s5,na.rm=TRUE)) %>%
  mutate(sdist_id=1:n()) %>%
  write_csv(file.path(DATA_PATH,TRIAL_DIR,"sdist.csv"))
