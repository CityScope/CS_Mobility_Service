library(TraMineR)

city='Detroit'
c_type="complete"
n_clusters=6
time_int=60

df_path=paste('cities/', city,'/clean/clust_mat_', time_int, '_', n_clusters, '_', c_type,'.csv', sep='')

act_seq=read.csv(df_path)
seq_len=length(names(act_seq))-2
print(paste('Length of sequence:', seq_len))

alph=seqstatl(act_seq[, seq(1,seq_len)])
print(paste('Alphabet:', toString(alph)))
act_seq.alphabet <- alph
act_seq.labels <- c('College', 'Drop-off', 'Eat', 'Groceries', 'Home', 
                    'Health', 'Recreation', 'Services', 'Visit', 'Work', 'Exercise', 'Religion')

act_seq.scodes <- alph
act_seq.seq <- seqdef(act_seq, 1:seq_len, alphabet = act_seq.alphabet, states = act_seq.scodes, 
                      labels = act_seq.labels, xtstep = 6)

cl1=act_seq$clust
cl1.fac <- factor(cl1, labels = paste("Cluster", 1:n_clusters))
# # Plot sequences
# par(mfrow = c(2, 2))
# seqiplot(act_seq.seq, with.legend = FALSE, border = NA)
# seqIplot(act_seq.seq, sortv = "from.start", with.legend = FALSE)
# seqfplot(act_seq.seq, with.legend = FALSE, border = NA)
# seqlegend(act_seq.seq)
# 
# # Descriptive Stats
# par(mfrow = c(2, 1))
# seqdplot(act_seq.seq, with.legend = FALSE, border = NA)
# # seqHtplot(act_seq.seq)
# # seqmsplot(act_seq.seq, with.legend = FALSE, border = NA)
# seqmtplot(act_seq.seq, with.legend = FALSE)

# Plot Clusters
# Plot all the sequences within each cluster.
# quartz()
# seqIplot(act_seq.seq, group = cl1.fac, sortv = "from.start")

# State Distribution in each cluster
quartz()
seqdplot(act_seq.seq, group = cl1.fac, border = NA)

# Representative Sequences of each cluster
quartz()
seqrplot(act_seq.seq, diss = dist.om1, group = cl1.fac,
         border = NA)