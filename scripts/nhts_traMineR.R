library(TraMineR)

city='Detroit'
act_seq=read.csv(paste('cities/', city,'/clean/seq_matrix_24.csv', sep=''))
seq_len=length(names(act_seq))-1
print(paste('Length of sequence:', seq_len))

alph=seqstatl(act_seq[, seq(1,seq_len)])
print(paste('Alphabet:', toString(alph)))
act_seq.alphabet <- alph
act_seq.labels <- c('College', 'Drop-off', 'Eat', 'Groceries', 'Home', 
                    'Health', 'Recreation', 'Services', 'Visit', 'Work', 'Exercise', 'Religion')
act_seq.scodes <- alph
act_seq.seq <- seqdef(act_seq, 1:seq_len, alphabet = act_seq.alphabet, states = act_seq.scodes, 
                      labels = act_seq.labels, xtstep = 6)
# Plot sequences
par(mfrow = c(2, 2))
seqiplot(act_seq.seq, with.legend = FALSE, border = NA)
seqIplot(act_seq.seq, sortv = "from.start", with.legend = FALSE)
seqfplot(act_seq.seq, with.legend = FALSE, border = NA)
seqlegend(act_seq.seq)

quartz()
seqIplot(act_seq.seq, sortv = "from.start", with.legend = TRUE)




# Descriptive Stats
par(mfrow = c(2, 1))
seqdplot(act_seq.seq, with.legend = FALSE, border = NA)
# seqHtplot(act_seq.seq)
# seqmsplot(act_seq.seq, with.legend = FALSE, border = NA)
seqmtplot(act_seq.seq, with.legend = FALSE)

# Clustering
dist.om1 <- seqdist(act_seq.seq, method = "OM", indel = 1, sm = "TRATE")

library(cluster)
clusterward1 <- agnes(dist.om1, diss = TRUE, method = "ward")
# plot(clusterward1, which.plot = 2)

n_clusters=7

cl1 <- cutree(clusterward1, k = n_clusters)
cl1.fac <- factor(cl1, labels = paste("Cluster", 1:n_clusters))

# Plot all the sequences within each cluster.
quartz()
seqIplot(act_seq.seq, group = cl1.fac, sortv = "from.start")

# State Distribution in each cluster
quartz()
seqdplot(act_seq.seq, group = cl1.fac, border = NA)

# Representative Sequences of each cluster
quartz()
seqrplot(act_seq.seq, diss = dist.om1, group = cl1.fac,
         border = NA)


# name rows
levels(cl1.fac)[levels(cl1.fac)=="Cluster 1"] <- "Homemakers"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 2"] <- "9-to-5ers"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 3"] <- "Afternoon Adventurers"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 4"] <- "Night-shifters"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 5"] <- "Socialites"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 6"] <- "Work-Life Balancers"
levels(cl1.fac)[levels(cl1.fac)=="Cluster 7"] <- "Students"
# Plot all the sequences within each cluster.
quartz()
seqIplot(act_seq.seq, group = cl1.fac, sortv = "from.start")

# sample 100 of each and save
act_seq$cluster=cl1
act_seq$cluster_name=cl1.fac
write.csv(act_seq,paste('cities/', city,'/clean/profile_labels.csv', sep=""))

all_cluster_names=levels(cl1.fac)
subset_cluster=act_seq[act_seq$cluster_name==all_cluster_names[1],]
motif_sample=subset_cluster[sample(nrow(subset_cluster), 20), ]
for (i in 2:length(all_cluster_names)){
  subset_cluster=act_seq[act_seq$cluster_name==all_cluster_names[i],]
  motif_sample=rbind(motif_sample, subset_cluster[sample(nrow(subset_cluster), 20), ])
}
motif_sample=motif_sample[,!names(motif_sample)=='id']
# plot samples
samples.seq <- seqdef(motif_sample, 1:seq_len, alphabet = act_seq.alphabet, states = act_seq.scodes, 
                      labels = act_seq.labels, xtstep = 6)
quartz()
seqIplot(samples.seq, group = motif_sample$cluster_name, sortv = "from.start")
write.csv(motif_sample, paste('cities/', city,'/clean/motif_samples.csv', sep=''), row.names=FALSE)
