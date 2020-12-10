library(tidyverse)
library(readr)

DATA_PATH<-"~/Dropbox (MIT)/batadal/small_batch"

res <- read_csv(file.path(DATA_PATH,"train_warmstart","final_results.csv"))
sdist <- read_csv(file.path(DATA_PATH,"sdist.csv"))

res <- res %>%
  separate(model_name,c("sdist_id","dataset")) %>%
  select(-trial) %>% 
  arrange(desc(test_acc)) 

res <- sdist %>%
  select(budget,fortify,trial_type,trial_param,sdist_id) %>%
  mutate(sdist_id=as.character(sdist_id)) %>%
  right_join(res,by="sdist_id") %>%
  mutate(trial_type=ifelse(is.na(trial_type),"unif_sdist",trial_type))

res <- res %>%
  mutate(sens=test_tp/(test_tp+test_fn),
         spec=test_tn/(test_tn+test_fp)) 
res %>%
  arrange(desc(sens)) %>% 
  select(budget,fortify,trial_type,trial_param,spec,sens) %>%
  rbind(data.frame(budget=NA,fortify=NA,trial_type="real",
                   trial_param=NA,spec=0.227,sens=0.933)) %>% 
  ggplot(aes(x=sens,y=spec,color=trial_type))  +
  geom_point(size=3) + theme_bw() +
  scale_color_discrete("Training data",
                       labels=c("bfs"="Synthetic - Targeted demand",
                                "real"="Real",
                                "unif"="Synthetic - Uniform demand",
                                "unif_sdist"="Synthetic - Uniform sensor")) +
  labs(x="Sensitivity (%)",y="Specificity (%)") +
  ggtitle("Sensitivity vs. Specificity on test data by training data") +
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title.y = element_text(size=12),
        title = element_text(size=16),
        legend.text = element_text(size=12),
        strip.text = element_text(size = 12))

res %>%
  filter(sens > 0.75) %>% 
  arrange(desc(spec)) %>% View()

res %>%
  arrange(desc(sens)) %>% 
  select(budget,fortify,trial_type,trial_param,spec,sens,test_acc) %>%
  rbind(data.frame(budget=NA,fortify=NA,trial_type="real",
                   trial_param=NA,spec=0.227,sens=0.933,test_acc=0.739)) %>% 
  ggplot(aes(x=sens,y=spec,color=fortify))  +
  geom_point(size=3) + theme_bw() +
  scale_color_discrete("",
                       labels=c("TRUE"="Fortified",
                                "FALSE"="Not fortified")) +
  labs(x="Sensitivity (%)",y="Specificity (%)") +
  ggtitle("Sensitivity vs. Specificity on test data by test accuracy") +
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title.y = element_text(size=12),
        title = element_text(size=16),
        legend.text = element_text(size=12),
        strip.text = element_text(size = 12))
