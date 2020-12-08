library(tidyverse)
library(readr)


DATA_PATH<-"/Users/kaylacummings/Dropbox (MIT)/batadal"
TRIAL_DIR<-"small_batch"
INTERDICTION_DIR<-"interdiction_viz"

dir.create(file.path(DATA_PATH,TRIAL_DIR,INTERDICTION_DIR))

trial_info<-read_csv(file.path(DATA_PATH,TRIAL_DIR,"trial_info.csv")) 
int<-read_csv(file.path(DATA_PATH,TRIAL_DIR,"interdicted.csv"))
results<-read_csv(file.path(DATA_PATH,TRIAL_DIR,"results.csv"))

int <- int %>%
  left_join(trial_info,by="trial_id") %>%
  group_by(edge_id,trial_type,trial_param) %>%
  summarise(num_int=n())

# there are 6 trial types - one CSV for each to visualize in networkx
groups <- int %>%
  group_by(trial_type,trial_param) %>%
  summarise(num=n()) %>%
  filter(trial_type!="full") %>%
  select(trial_type,trial_param)
for (i in 1:nrow(groups)) {
  trial_param <- groups$trial_param[i]
  trial_type <- groups$trial_type[i]
  int %>%
    filter(trial_param==trial_param) %>%
    write_csv(file.path(DATA_PATH,TRIAL_DIR,INTERDICTION_DIR,
                        paste0("network","_",trial_type,"_",trial_param,".csv")))
}


# results of interdiction model - max flows by budget, fortification
baseline <- results %>%
  filter(budget==0) %>%
  select(trial_id,baseline_obj=obj) %>%
  unique()
results <- results %>%
  filter(budget!=0) %>%
  left_join(baseline,by=c("trial_id")) %>%
  mutate(s1=as.numeric(s1),
         s2=as.numeric(s2),
         s3=as.numeric(s3),
         s4=as.numeric(s4),
         s5=as.numeric(s5))

results %>% 
  filter(trial_type=="bfs") %>%
  mutate(pct_dem=obj/baseline_obj) %>% 
  ggplot() +
  geom_violin(aes(x=budget,y=pct_dem,fill=fortify)) +
  facet_wrap(~trial_param) +
  ylim(c(0,1))
