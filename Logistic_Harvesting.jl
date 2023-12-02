using Plots, DifferentialEquations
using Interpolations, Random, Distributions
gr()
#Define parameters to generate data
a=zeros(3)
λ=0.01; d=0.002; K=100.0; t=0:100:1000; σ=5.0;
tt=0:1:1000;

function model(t,a)
y=zeros(length(t))
    C0=5
gg = a[1]-a[2]
KK = a[3]*gg/a[1]
#Exact solution
c(t) = KK*C0/(C0+(KK-C0)*exp(-gg*t))
#Exact solution if net growth is zero
if gg == 0
    for i in 1:length(t) 
        y[i] = C0
        end
    else

for i in 1:length(t) 
y[i] = c(t[i])
end
end 

return y
end

data0=zeros(length(t));
data=zeros(length(t));
#data0=model(t,[λ,d,K]);
#data=data0+σ*randn(length(t));

#data realisation obtained by solving the model and adding Gaussian noise
data=[5
12.824165748778066
 18.961389706231145
 33.8143356717012
 47.01659003146925
 50.366643562660286
 71.46405287094763
 80.78683251674342
 65.2856106305211
 84.84773051201819
 77.54269425909838];

#Function to evaluate the loglikelihood 
function loglhood(data,a)
    σ=5
    y=zeros(length(t))
    y=model(t,a);
    e=0;
    dist=Normal(0,σ);
    e=loglikelihood(dist,data-y) 
    return sum(e)
end

#Define simple parameter bounds, these are user-specified and chosen to comfortably contain the true parameters
λmin=0.0001
λmax=0.05
dmin=0.0
dmax=0.01
Kmin=50
Kmax=200

#Set-up Q^3 cube of points to evaluate the log-likelihood
Q=500
λg=LinRange(λmin,λmax,Q);
dg=LinRange(dmin,dmax,Q);
Kg=LinRange(Kmin,Kmax,Q);

ll=zeros(Q,Q,Q);
lln=zeros(Q,Q,Q);


for i in 1:Q
    for j in 1:Q
        for k in 1:Q
        ll[i,j,k] = loglhood(data,[λg[i],dg[j],Kg[k]])
        end
    end
end

#Identify the maximum loglikelihood, and normalise
(llmax,Index) = findmax(ll);
lln = ll.-llmax;

#Identify and store the MLE
λmle = λg[Index[1]]
dmle = dg[Index[2]]
Kmle = Kg[Index[3]]


ymle=zeros(length(tt))
ymle=model(tt,[λmle,dmle,Kmle])
p1=plot(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
p1=scatter!(t,data,legend=false,msw=0,ms=7,color=:darkorange,msa=:darkorange)
display(p1)
savefig(p1, "mle.pdf")


#Make plots showing slices through the full loglikelihood function, superimpose the 95% threshold.  This is a useful visualisation tool
df=3
llstar=-quantile(Chisq(df),0.95)/2;

p1=contourf(λg,dg,lln[:,:,240],lw=0,xlabel="r",ylabel="d",title="K=K1",c=:greens,colorbar=false)
p1=contour!(λg,dg,lln[:,:,240],levels=[llstar],lw=4,c=:red,legend=false)

p2=contourf(λg,dg,lln[:,:,245],lw=0,xlabel="λ",ylabel="d",title="K=K2",c=:greens,colorbar=false)
p2=contour!(λg,dg,lln[:,:,245],levels=[llstar],lw=4,c=:red,legend=false)

p3=contourf(λg,dg,lln[:,:,255],lw=0,xlabel="λ",ylabel="d",title="K=K3",c=:greens,colorbar=false)
p3=contour!(λg,dg,lln[:,:,255],levels=[llstar],lw=4,c=:red,legend=false)

p4=contourf(λg,dg,lln[:,:,260],lw=0,xlabel="λ",ylabel="d",title="K=K4",c=:greens,colorbar=false)
p4=contour!(λg,dg,lln[:,:,260],levels=[llstar],lw=4,c=:red,legend=false)

clims=extrema(lln)
h2 = scatter([0,0], [0,0], zcolor=[0,3], ms=0, clims=clims, xlims=(1,1.1), label="", c=:greens, framestyle=:none, right_margin=5Plots.mm)
l = @layout [grid(2, 2) a{0.05w}]
p_all = plot(p1, p2, p3, p4, h2, layout=l, link=:all)
savefig(p_all, "Fulllikelihood.pdf")


#count points on the discretised cube where the loglikelihood exceeds the 95% threshold
df=3
llstar=-quantile(Chisq(df),0.95)/2;
count = 0
for i in 1:Q
    for j in 1:Q
        for k in 1:Q
        if lln[i,j,k] >= llstar
            count+=1
        end
        end
    end
end

λsampled=zeros(count)
dsampled=zeros(count)
Ksampled=zeros(count)

count = 0
for i in 1:Q
    for j in 1:Q
        for k in 1:Q
        if lln[i,j,k] >= llstar
        count +=1
        λsampled[count] = λg[i]
        dsampled[count] = dg[j]
        Ksampled[count] = Kg[k]
        end
        end
    end
end
#Compute the upper and lower traces 
upperfl=zeros(length(tt))
lowerfl=1000*ones(length(tt))
trace=zeros(length(tt))

for i in 1:count
trace = model(tt,[λsampled[i],dsampled[i],Ksampled[i]])

  for k in 1:length(tt)
     if trace[k] >= upperfl[k]
         upperfl[k]=trace[k]
     end

     if trace[k] <= lowerfl[k]
         lowerfl[k] = trace[k]
     end
  end

end
#Plot the confidence interval from the full likelihood function 
q1=plot(tt,lowerfl,lw=0,fillrange=upperfl,fillalpha=0.40,color=:gold,label=false,xlims=(0,maximum(tt)))
p1=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
q1=scatter!(t,data,legend=false,msw=0,ms=5,color=:darkorange,msa=:darkorange)
savefig(q1,"Fulllikelihood_Prediction.pdf")

#Set up variables to store the univariate profile likelihood
uvλ=zeros(Q)
uvd=zeros(Q)
uvK=zeros(Q)

#Optimise out using maximisation across the grid
for i in 1:Q
   uvλ[i] = maximum(lln[i,:,:])
   uvd[i] = maximum(lln[:,i,:])
   uvK[i] = maximum(lln[:,:,i])
end

df=1
llstar=-quantile(Chisq(df),0.95)/2;

r1=plot(λg,uvλ,lw=4,xlabel="λ",ylabel="ℓp",ylims=(-3,0.05),legend=false)
r1=hline!([llstar],lw=4)
r1=vline!([λmle],lw=4)
r2=plot(dg,uvd,lw=4,xlabel="d",ylabel="ℓp",ylims=(-3,0.05),legend=false)
r2=hline!([llstar],lw=4)
r2=vline!([dmle],lw=4)
r3=plot(Kg,uvK,lw=4,xlabel="K",ylabel="ℓp",ylims=(-3,0.05),legend=false)
r3=hline!([llstar],lw=4)
r3=vline!([Kmle],lw=4)
r4=plot(r1,r2,r3,layout=(3,1))
savefig(r4, "Univariate.pdf")


#Set up to store the bivariate profile likelihood
bvλd=zeros(Q,Q);
bvnλd=zeros(Q,Q);
bvλK=zeros(Q,Q);
bvnλK=zeros(Q,Q);
bvdK=zeros(Q,Q);
bvndK=zeros(Q,Q);

#Maximise along the grid for the lambda-d, lambda-K, and d-K bivariate profile likelihoods
for i in 1:Q
    for j in 1:Q
    (bvλd[i,j],Index) = findmax(lln[i,j,:])
    bvnλd[i,j]=Kg[Index[1]]
    (bvλK[i,j],Index) = findmax(lln[i,:,j])
    bvnλK[i,j]=dg[Index[1]]
    (bvdK[i,j],Index) = findmax(lln[:,i,j])
    bvndK[i,j]=λg[Index[1]]
    end
end

df=2
llstar=-quantile(Chisq(df),0.95)/2;
#Plot the bivariate profile likelihood
q1=contourf(λg,dg,bvλd',lw=0,xlabel="λ",ylabel="d",c=:blues)
q1=contour!(λg,dg,bvλd',levels=[llstar],lw=4,c=:red)
#q2=contourf(λg,dg,bvnλd,lw=0,xlabel="λ",ylabel="d",c=:reds)
#q22=plot(q1,q2,layout=(1,2))
savefig(q1,"Bivariateλd.pdf")

#Compute the prediction intervals from the lambda-d bivariate profile likelihood
count = 0
for i in 1:Q
    for j in 1:Q
        if bvλd[i,j] >= llstar
            count+=1
        end
    end
end


λsampled=zeros(count)
dsampled=zeros(count)
Ksampled=zeros(count)

count = 0
for i in 1:Q
    for j in 1:Q
        if bvλd[i,j] >= llstar
        count +=1
        λsampled[count] = λg[i]
        dsampled[count] = dg[j]
        Ksampled[count] = bvnλd[i,j]
        end
    end
end



upperbvλd=zeros(length(tt))
lowerbvλd=1000*ones(length(tt))
trace=zeros(length(tt))

for i in 1:count
trace = model(tt,[λsampled[i],dsampled[i],Ksampled[i]])

  for k in 1:length(tt)
     if trace[k] >= upperbvλd[k]
         upperbvλd[k]=trace[k]
     end

     if trace[k] <= lowerbvλd[k]
         lowerbvλd[k] = trace[k]
     end
  end

end

q3=plot(tt,lowerbvλd,lw=0,fillrange=upperbvλd,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
q3=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
savefig(q3,"Bivariatepredictionsλd.pdf")






q2=contourf(λg,Kg,bvλK',lw=0,xlabel="λ",ylabel="K",c=:blues)
q2=contour!(λg,Kg,bvλK',levels=[llstar],lw=4,c=:red,legend=false)
#q22=contourf(λg,Kg,bvnλK,lw=0,xlabel="λ",ylabel="K",c=:reds)
#q20=plot(q2,q22,layout=(1,2))
savefig(q2,"BivariateλK.pdf")

#Compute the prediction intervals from the lambda-K bivariate profile likelihood
count = 0
for i in 1:Q
    for j in 1:Q
        if bvλK[i,j] >= llstar
            count+=1
        end
    end
end


λsampled=zeros(count)
dsampled=zeros(count)
Ksampled=zeros(count)

count = 0
for i in 1:Q
    for j in 1:Q
        if bvλK[i,j] >= llstar
        count +=1
        λsampled[count] = λg[i]
        Ksampled[count] = Kg[j]
        dsampled[count] = bvnλK[i,j]
        end
    end
end



upperbvλK=zeros(length(tt))
lowerbvλK=1000*ones(length(tt))
trace=zeros(length(tt))

for i in 1:count
trace = model(tt,[λsampled[i],dsampled[i],Ksampled[i]])

  for k in 1:length(tt)
     if trace[k] >= upperbvλK[k]
         upperbvλK[k]=trace[k]
     end

     if trace[k] <= lowerbvλK[k]
         lowerbvλK[k] = trace[k]
     end
  end

end

qq2=plot(tt,lowerbvλK,lw=0,fillrange=upperbvλK,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
qq2=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
savefig(qq2,"BivariatepredictionsλK.pdf")






q3=contourf(dg,Kg,bvdK',lw=0,xlabel="d",ylabel="K",c=:blues)
q3=contour!(dg,Kg,bvdK',levels=[llstar],lw=4,c=:red,legend=false)
#q32=contourf(dg,Kg,bvndK,lw=0,xlabel="d",ylabel="K",c=:reds)
#q30=plot(q3,q32,layout=(1,2))
savefig(q3,"BivariatedK.pdf")


#Compute the prediction intervals from the d-K bivariate profile likelihood
count = 0
for i in 1:Q
    for j in 1:Q
        if bvdK[i,j] >= llstar
            count+=1
        end
    end
end


λsampled=zeros(count)
dsampled=zeros(count)
Ksampled=zeros(count)

count = 0
for i in 1:Q
    for j in 1:Q
        if bvdK[i,j] >= llstar
        count +=1
        dsampled[count] = dg[i]
        Ksampled[count] = Kg[j]
        λsampled[count] = bvndK[i,j]
        end
    end
end



tt=0:1:maximum(t)
upperbvdK=zeros(length(tt))
lowerbvdK=1000*ones(length(tt))
trace=zeros(length(tt))

for i in 1:count
trace = model(tt,[λsampled[i],dsampled[i],Ksampled[i]])

  for k in 1:length(tt)
     if trace[k] >= upperbvdK[k]
         upperbvdK[k]=trace[k]
     end

     if trace[k] <= lowerbvdK[k]
         lowerbvdK[k] = trace[k]
     end
  end

end

qq3=plot(tt,lowerbvdK,lw=0,fillrange=upperbvdK,fillalpha=0.40,color=:purple,label=false,xlims=(0,tt))
qq3=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
savefig(qq3,"BivariatepredictionsdK.pdf")

#Compute the union of bivariate profiles to form the approximate prediction interval
upperbv=zeros(length(tt))
lowerbv=zeros(length(tt))
for i in 1:length(tt)
upperbv[i] = max(upperbvλd[i],upperbvλK[i],upperbvdK[i])
lowerbv[i] = min(lowerbvλd[i],lowerbvλK[i],lowerbvdK[i])
end

qq4=plot(tt,lowerbv,lw=0,fillrange=upperbv,fillalpha=0.40,color=:purple,label=false,xlims=(0,tt))
qq4=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])




q1=plot(tt,lowerfl,lw=0,fillrange=upperfl,fillalpha=0.40,color=:gold,label=false,xlims=(0,tt))
q1=plot!(tt,lowerbv,lw=3,color=:purple,ls=:dash,legend=false)
q1=plot!(tt,upperbv,lw=3,color=:purple,ls=:dash,legend=false)
q1=plot!(tt,ymle,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,100),xticks=[0,500,1000],yticks=[0,50,100])
savefig(q1,"PredictionComparison.pdf")