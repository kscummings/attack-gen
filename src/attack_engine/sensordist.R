library(tidyverse)
library(readr)

DATA_PATH<-"/Users/kaylacummings/Dropbox (MIT)/batadal"
TRIAL_DIR<-"test_13756"

res<-read_csv(file.path(DATA_PATH,TRIAL_DIR,"results.csv"))
res %>%
  group_by(budget,fortify,trial_type,trial_param) %>%
  summarise(s1=sum(s1,na.rm=TRUE),
            s2=sum(s2,na.rm=TRUE),
            s3=sum(s3,na.rm=TRUE),
            s4=sum(s4,na.rm=TRUE),
            s5=sum(s5,na.rm=TRUE)) %>%
  write_csv(file.path(DATA_PATH,TRIAL_DIR,"sdist.csv"))
