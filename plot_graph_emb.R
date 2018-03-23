library('data.table')
library('igraph')
library('ggnetwork')
library('ggplot2')

NUM_EMB <- 100

careers <- fread('../ubuntu-corpus/careers.txt')
setnames(careers, names(careers), c('user', 'day', 'i'))
careers <- careers[,.(num_days=nrow(.SD), first_day=min(day)),by=.(user)]

emb <- fread('graph_emb.txt')
setnames(emb, c('id', 'user', paste0('dim', 1:NUM_EMB)))
setkey(emb, 'user')
setkey(careers, 'user')
emb <- merge(emb, careers)
write.table(emb, file='careers_emb.best', sep='\t', quote=F, row.names=F, col.names=T)

for (i in 1:NUM_EMB) {
    dim <- emb[, sprintf('dim%d', i), with=F]
    dim <- dim ^ 2

    set(emb, j=sprintf('dim_%d_sq', i), value=dim)
}
formula <- paste0(sprintf("dim_%d_sq+dim%d", 1:NUM_EMB, 1:NUM_EMB), collapse='+')
model <- lm(paste0('log(num_days)~', formula, collapse='+'), data=emb)
print(summary(model))

print(head(emb))

smr <- summary(model)$coef
dims <- rownames(smr)[order(abs(smr[,1]) * -1)]
dims <- dims[dims != '(Intercept)' & !grepl('sq', dims)]
dims <- dims[2:3]

emb[, num_days_bucket:=floor(log(num_days))]
sampling.weights <- 1 / prop.table(table(emb$num_days_bucket))
sampling.weights <- sampling.weights / sum(sampling.weights)
prob <- sampling.weights[as.character(emb$num_days_bucket)]
emb_plot <- emb[sample(1:nrow(emb), 10000, prob=prob), ]

p <- ggplot(aes_string(
    x=sprintf('log(%s - min(%s))', dims[1], dims[1]),
    y=sprintf('log(%s - min(%s))', dims[2], dims[1]),
    color='log(num_days+1)'), data=emb_plot)
p <- p + geom_point(alpha=0.8)
p <- p + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=3)
print(p)


p13 <- ggplot(aes(x=dim1, y=dim3, color=log(num_days+1)), data=emb_plot)
p13 <- p13 + geom_point(alpha=0.8)
p13 <- p13 + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=3)

p12 <- ggplot(aes(x=dim1, y=dim2, color=log(num_days+1)), data=emb_plot)
p12 <- p12 + geom_point(alpha=0.8)
p12 <- p12 + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=3)

p23 <- ggplot(aes(x=dim2, y=dim3, color=log(num_days+1)), data=emb_plot)
p23 <- p23 + geom_point(alpha=0.8)
p23 <- p23 + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=3)

library('gridExtra')
grid.arrange(p12, p13, p23)

df <- fread('df_tsne.tsv')
setnames(df, 'label', 'user')
setkey(df, 'user')
setkey(careers, 'user')
df <- merge(df, careers)

setnames(df, c('x-tsne', 'y-tsne'), c('x', 'y'))
p <- ggplot(data=df)
p <- p + geom_point(aes(x=x, y=y, color=log(num_days, base=2)))
p <- p + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=3)
#p <- p + xlim(c(-10, 10))
#p <- p + ylim(c(-10, 10))
p <- p + guides(color=guide_legend(title='Log2 Num Days on Site'))
p <- p + theme(legend.position='bottom')
png('social_graph_tsne.png')
print(p)
dev.off()
