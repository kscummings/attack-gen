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
  tparam <- groups$trial_param[i]
  ttype <- groups$trial_type[i]
  int %>%
    filter(trial_param==tparam) %>%
    write_csv(file.path(DATA_PATH,TRIAL_DIR,INTERDICTION_DIR,
                        paste0("network","_",ttype,"_",tparam,".csv")))
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
  geom_boxplot(aes(x=as.factor(budget),y=1-pct_dem,fill=fortify)) +
  facet_wrap(~trial_param) +
  ylim(c(0,1))+
  theme_bw()+
  labs(x="Interdiction budget (number of links)",
       y="Percent decrease") +
  scale_fill_discrete("Fortified") +
  ggtitle("Percent decrease in maximum flow",
          subtitle="Cluster targeting strategy - stratified by BFS depth")+
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title.y = element_text(size=12),
        title = element_text(size=16),
        legend.text = element_text(size=12),
        strip.text = element_text(size = 12))



# look at distributions 
results %>%
  filter(trial_type=="bfs") %>%
  group_by(budget,fortify,trial_param) %>%
  summarise(total=sum(s1+s2+s3+s4+s5),
            s1=sum(s1)/total,
            s2=sum(s2)/total,
            s3=sum(s3)/total,
            s4=sum(s4)/total,
            s5=sum(s5)/total) %>%
  pivot_longer(c(s1,s2,s3,s4,s5),
               values_to="int_prop",
               names_to="sensor_set") %>%
  ggplot(aes(x=sensor_set,y=int_prop,fill=fortify)) +
  geom_bar(stat="identity",position="dodge") +
  facet_grid(trial_param ~ as.factor(budget)) +
  theme_bw() +
  labs(x="Sensor set",y="Interdiction proportion") +
  scale_fill_discrete("Fortify") 
results %>%
  filter(trial_type=="unif") %>%
  group_by(budget,fortify,trial_param) %>%
  summarise(total=sum(s1+s2+s3+s4+s5),
            s1=sum(s1)/total,
            s2=sum(s2)/total,
            s3=sum(s3)/total,
            s4=sum(s4)/total,
            s5=sum(s5)/total) %>%
  pivot_longer(c(s1,s2,s3,s4,s5),
               values_to="int_prop",
               names_to="sensor_set") %>%
  ggplot(aes(x=sensor_set,y=int_prop,fill=fortify)) +
  geom_bar(stat="identity",position="dodge") +
  facet_grid(trial_param ~ as.factor(budget)) +
  theme_bw() +
  labs(x="Sensor set",y="Interdiction proportion") +
  scale_fill_discrete("Fortify")
