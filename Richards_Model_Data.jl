using Plots
using Interpolations, Random, Distributions
using Roots, NLopt
gr()
a=zeros(4)
λ =0.005; β=1.0; K=80.0; C0=1.0;
tt=0:1:4048;
t=[0, 769,1140,1488,1876,2233,2602,2889,3213,3621,4028];
data=[2.352254642,4.396074415,8.434146341,22.25079365,38.9,59.04803013,67.84648814,69.51641791,74.09765494,82.29230769,80.88291457]



function model(t,a)
y=zeros(length(t))
c(t) = a[3]*a[4]/(a[4]^a[2]+(a[3]^a[2]-a[4]^a[2])*exp(-a[1]*a[2]*t))^(1/a[2])
for i in 1:length(t) 
y[i] = c(t[i])
end
return y
end


function loglhood(data,a)
    y=zeros(length(t))
    y=model(t,a);
    e=0;
    dist=Normal(0,2);
    e=loglikelihood(dist,data-y) 
    return sum(e)
end

#Section 7: Define simple parameter bounds,
λmin=0.0001
λmax=0.02
Kmin=60
Kmax=100
C0min=0.0001
C0max=5
βmin = 0.001
βmax = 5.0


function Optimise(fun,θ₀,lb,ub)
    
    tomax=(θ,∂θ)->fun(θ)
    opt=Opt(:LN_BOBYQA,length(θ₀))
    opt.max_objective=tomax
    opt.lower_bounds=lb      
    opt.upper_bounds=ub
    opt.maxtime=1*60
    res = optimize(opt,θ₀)
    return res[[2,1]]

end



#Section 8: Function to be optimised for MLE
a=zeros(4)
function funmle(a)
return loglhood(data,a)
end

#Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
θG = [λ,β,K,C0]
lb=[λmin,βmin,Kmin,C0min]
ub=[λmax,βmax,Kmax,C0max]
(xopt,fopt)  = Optimise(funmle,θG,lb,ub)
fmle=fopt
λmle=xopt[1]; 
βmle = xopt[2];
Kmle=xopt[3]; 
C0mle=xopt[4];
ymle(t) = Kmle*C0mle/(C0mle^βmle+(Kmle^βmle-C0mle^βmle)*exp(-βmle*λmle*t))^(1/βmle);
p1=plot(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
p1=scatter!(t,data,legend=false,msw=0,ms=7,color=:darkorange,msa=:darkorange)
display(p1)
savefig(p1, "mle.pdf")



df=4
llstar=-quantile(Chisq(df),0.95)/2

λmin=0.0001
λmax=0.02
Kmin=75
Kmax=90
C0min=0.001
C0max=2.0
βmin=0.001
βmax=3.0


M=1000
λsampled=zeros(M)
βsampled=zeros(M)
Ksampled=zeros(M)
C0sampled=zeros(M)
lls=zeros(M)
kount = 0

while kount < M
λg=rand(Uniform(λmin,λmax))
βg=rand(Uniform(βmin,βmax))
Kg=rand(Uniform(Kmin,Kmax))
C0g=rand(Uniform(C0min,C0max))
    if (loglhood(data,[λg,βg,Kg,C0g])-fmle) >= llstar
    kount+=1
    println(kount)
    lls[kount]=loglhood(data,[λg,βg,Kg,C0g])-fmle
    λsampled[kount]=λg;
    βsampled[kount]=βg;
    Ksampled[kount]=Kg;
    C0sampled[kount]=C0g;
    end
end

q1=scatter(λsampled,legend=false)
q1=hline!([λmin,λmax],legend=false)

q2=scatter(βsampled,legend=false)
q2=hline!([βmin,βmax],legend=false)

q3=scatter(Ksampled,legend=false)
q3=hline!([Kmin,Kmax],legend=false)

q4=scatter(C0sampled,legend=false)
q4=hline!([C0min,C0max],legend=false)


CtraceF = zeros(length(tt),M)
CUF=zeros(length(tt))
CLF=zeros(length(tt))

for i in 1:M
CtraceF[:,i]=model(tt,[λsampled[i],βsampled[i],Ksampled[i],C0sampled[i]])
end


#Define point-wise maximum/minimum
for i in 1:length(tt)
CUF[i] = maximum(CtraceF[i,:])
CLF[i] = minimum(CtraceF[i,:])
end

q1=plot(tt,CLF,lw=0,fillrange=CUF,fillalpha=0.40,color=:gold,label=false,xlims=(0,maximum(tt)))
q1=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
q1=scatter!(t,data,legend=false,msw=0,ms=5,color=:darkorange,msa=:darkorange)
savefig(q1,"Fulllikelihood_Prediction.pdf")


#1 bivriate(λ,β)    
df=2
llstar=-quantile(Chisq(df),0.95)/2
function bivariateλβ(λ,β)
 function funλβ(a)
    return loglhood(data,[λ,β,a[1],a[2]])
    end
    θG = [Kmle,C0mle]
    lb=[Kmin,C0min]
    ub=[Kmax,C0max]
    (xopt,fopt)  = Optimise(funλβ,θG,lb,ub)
llb=fopt-fmle
return llb,xopt
end 
f(x,y) = bivariateλβ(x,y)
g(x,y)=f(x,y)[1]-llstar

#Define small parameter on the scale of parameter β
ϵ=(βmax-βmin)/10^5
N=250
λsamples=zeros(2*N)
βsamples=zeros(2*N)
Ksamples=zeros(2*N)
C0samples=zeros(2*N)
count=0

#Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
while count < N
x=rand(Uniform(λmin,λmax))
y0=rand(Uniform(βmin,βmax))
y1=rand(Uniform(βmin,βmax))
#If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
#vertical line separating the two points
if g(x,y0)*g(x,y1) < 0 
count+=1
println(count)
while abs(y1-y0) > ϵ && y1 < βmax && y1 > βmin
y2=(y1+y0)/2;
    if g(x,y0)*g(x,y2) < 0 
    y1=y2
    else
    y0=y2
    end


end

λsamples[count]=x;
βsamples[count]=y1;
Ksamples[count]=f(x,y1)[2][1]
C0samples[count]=f(x,y1)[2][2]
end
end 

#Define small number on the scale of the parameter λ
ϵ=(λmax-λmin)/10^2
count=0
while count < N
y=rand(Uniform(βmin,βmax))
x0=rand(Uniform(λmin,λmax))
x1=rand(Uniform(λmin,λmax))
#If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
#horizontal line separating the two points    
if g(x0,y)*g(x1,y) < 0 
count+=1
println(count)

while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
    x2=(x1+x0)/2;
        if g(x0,y)*g(x2,y) < 0 
        x1=x2
        else
        x0=x2
        end
    
    
    end


    λsamples[N+count]=x1;
    βsamples[N+count]=y;
    Ksamples[N+count]=f(x1,y)[2][1]
    C0samples[N+count]=f(x1,y)[2][2]
    end
    end 
#Plot the MLE and the 2N points identified on the boundary
aa1=scatter([λmle],[βmle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="β",label=false)
display(aa1)
for i in 1:2*N
aa1=scatter!([λsamples[i]],[βsamples[i]],markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
end
display(aa1)
savefig(aa1,"Bivariateλβ.pdf")
#Solve the model using the parameter values on the boundary of the bivariate profile
Ctrace1 = zeros(length(tt),2*N)
CU1=zeros(length(tt))
CL1=zeros(length(tt))
for i in 1:2*N
Ctrace1[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
end
    
#Calculate the maximum/minimum envelope of the solutions    
for i in 1:length(tt)
CU1[i] = maximum(Ctrace1[i,:])
CL1[i] = minimum(Ctrace1[i,:])
end
    


#Plot the family of solutions, the maximum/minimum envelope and the MLE
q1=plot(tt,CL1,lw=0,fillrange=CU1,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
q1=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
savefig(q1,"Bivariatepredictionsλβ.pdf")


#2 bivriate(λ,K)    
function bivariateλK(λ,K)
 function funλK(a)
    return loglhood(data,[λ,a[1],K,a[2]])
    end
    θG = [βmle,C0mle]
    lb=[βmin,C0min]
    ub=[βmax,C0max]
    (xopt,fopt)  = Optimise(funλK,θG,lb,ub)
llb=fopt-fmle
return llb,xopt
end 
f(x,y) = bivariateλK(x,y)
g(x,y)=f(x,y)[1]-llstar



#Define small parameter on the scale of parameter K
ϵ=(Kmax-Kmin)/10^5
N=250
λsamples=zeros(2*N)
βsamples=zeros(2*N)
Ksamples=zeros(2*N)
C0samples=zeros(2*N)
count=0

#Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
while count < N
x=rand(Uniform(λmin,λmax))
y0=rand(Uniform(Kmin,Kmax))
y1=rand(Uniform(Kmin,Kmax))
#If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
#vertical line separating the two points
if g(x,y0)*g(x,y1) < 0 
count+=1
println(count)
while abs(y1-y0) > ϵ && y1 < Kmax && y1 > Kmin
y2=(y1+y0)/2;
    if g(x,y0)*g(x,y2) < 0 
    y1=y2
    else
    y0=y2
    end


end

λsamples[count]=x;
Ksamples[count]=y1;
βsamples[count]=f(x,y1)[2][1]
C0samples[count]=f(x,y1)[2][2]
end
end 

#Define small number on the scale of the parameter λ
ϵ=(λmax-λmin)/10^2
count=0
while count < N
y=rand(Uniform(Kmin,Kmax))
x0=rand(Uniform(λmin,λmax))
x1=rand(Uniform(λmin,λmax))
#If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
#horizontal line separating the two points    
if g(x0,y)*g(x1,y) < 0 
count+=1
println(count)

while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
    x2=(x1+x0)/2;
        if g(x0,y)*g(x2,y) < 0 
        x1=x2
        else
        x0=x2
        end
    
    
    end


    λsamples[N+count]=x1;
    Ksamples[N+count]=y;
    βsamples[N+count]=f(x1,y)[2][1]
    C0samples[N+count]=f(x1,y)[2][2]
    end
    end 
#Plot the MLE and the 2N points identified on the boundary
aa2=scatter([λmle],[Kmle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="β",label=false)
display(aa2)
for i in 1:2*N
aa2=scatter!([λsamples[i]],[Ksamples[i]],markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
end
display(aa2)
savefig(aa2,"BivariateλK.pdf")
#Solve the model using the parameter values on the boundary of the bivariate profile
Ctrace2 = zeros(length(tt),2*N)
CU2=zeros(length(tt))
CL2=zeros(length(tt))
for i in 1:2*N
Ctrace2[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
end
    
#Calculate the maximum/minimum envelope of the solutions    
for i in 1:length(tt)
CU2[i] = maximum(Ctrace2[i,:])
CL2[i] = minimum(Ctrace2[i,:])
end
    


#Plot the family of solutions, the maximum/minimum envelope and the MLE
q2=plot(tt,CL2,lw=0,fillrange=CU2,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
q2=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
savefig(q2,"BivariatepredictionsλK.pdf")

#3 bivriate(λ,C0)    
function bivariateλC0(λ,C0)
    function funλC0(a)
       return loglhood(data,[λ,a[1],a[2],C0])
       end
       θG = [βmle,Kmle]
       lb=[βmin,Kmin]
       ub=[βmax,Kmax]
       (xopt,fopt)  = Optimise(funλC0,θG,lb,ub)
   llb=fopt-fmle
   return llb,xopt
   end 
   f(x,y) = bivariateλC0(x,y)
   g(x,y)=f(x,y)[1]-llstar
   
   
   
   #Define small parameter on the scale of parameter C0
   ϵ=(C0max-C0min)/10^4
   N=250
   λsamples=zeros(2*N)
   βsamples=zeros(2*N)
   Ksamples=zeros(2*N)
   C0samples=zeros(2*N)
   count=0
   
   #Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
   while count < N
   x=rand(Uniform(λmin,λmax))
   y0=rand(Uniform(C0min,C0max))
   y1=rand(Uniform(C0min,C0max))
   #If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #vertical line separating the two points
   if g(x,y0)*g(x,y1) < 0 
   count+=1
   println(count)
   while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
   y2=(y1+y0)/2;
       if g(x,y0)*g(x,y2) < 0 
       y1=y2
       else
       y0=y2
       end
   
   
   end
   
   λsamples[count]=x;
   C0samples[count]=y1;
   βsamples[count]=f(x,y1)[2][1]
   Ksamples[count]=f(x,y1)[2][2]
   end
   end 
   
   #Define small number on the scale of the parameter λ
   ϵ=(λmax-λmin)/10^2
   count=0
   while count < N
   y=rand(Uniform(C0min,C0max))
   x0=rand(Uniform(λmin,λmax))
   x1=rand(Uniform(λmin,λmax))
   #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #horizontal line separating the two points    
   if g(x0,y)*g(x1,y) < 0 
   count+=1
   println(count)
   
   while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
       x2=(x1+x0)/2;
           if g(x0,y)*g(x2,y) < 0 
           x1=x2
           else
           x0=x2
           end
       
       
       end
   
   
       λsamples[N+count]=x1;
       C0samples[N+count]=y;
       βsamples[N+count]=f(x1,y)[2][1]
       Ksamples[N+count]=f(x1,y)[2][2]
       end
       end 
   #Plot the MLE and the 2N points identified on the boundary
   aa3=scatter([λmle],[C0mle],xlims=(λmin,λmax),ylims=(0,2),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="C(0)",label=false)
   display(aa3)
   for i in 1:2*N
   aa3=scatter!([λsamples[i]],[C0samples[i]],xlims=(λmin,λmax),ylims=(0,2),markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
   end
   display(aa3)
   savefig(aa3,"BivariateλC0.pdf")
   #Solve the model using the parameter values on the boundary of the bivariate profile
   Ctrace3 = zeros(length(tt),2*N)
   CU3=zeros(length(tt))
   CL3=zeros(length(tt))
   for i in 1:2*N
   Ctrace3[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
   end
       
   #Calculate the maximum/minimum envelope of the solutions    
   for i in 1:length(tt)
   CU3[i] = maximum(Ctrace3[i,:])
   CL3[i] = minimum(Ctrace3[i,:])
   end
       
   
   #Plot the family of solutions, the maximum/minimum envelope and the MLE
   q3=plot(tt,CL3,lw=0,fillrange=CU3,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
   q3=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
   savefig(q3,"BivariatepredictionsλC0.pdf")

#4 bivriate(β,K)    
function bivariateβK(β,K)
    function funβK(a)
       return loglhood(data,[a[1],β,K,a[2]])
       end
       θG = [λmle,C0mle]
       lb=[λmin,C0min]
       ub=[λmax,C0max]
       (xopt,fopt)  = Optimise(funβK,θG,lb,ub)
   llb=fopt-fmle
   return llb,xopt
   end 
   f(x,y) = bivariateβK(x,y)
   g(x,y)=f(x,y)[1]-llstar
   
   
   
   #Define small parameter on the scale of parameter C0
   ϵ=(Kmax-Kmin)/10^5
   N=250
   λsamples=zeros(2*N)
   βsamples=zeros(2*N)
   Ksamples=zeros(2*N)
   C0samples=zeros(2*N)
   count=0
   
   #Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
   while count < N
   x=rand(Uniform(βmin,βmax))
   y0=rand(Uniform(Kmin,Kmax))
   y1=rand(Uniform(Kmin,Kmax))
   #If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #vertical line separating the two points
   if g(x,y0)*g(x,y1) < 0 
   count+=1
   println(count)
   while abs(y1-y0) > ϵ && y1 < Kmax && y1 > Kmin
   y2=(y1+y0)/2;
       if g(x,y0)*g(x,y2) < 0 
       y1=y2
       else
       y0=y2
       end
   
   
   end
   
   βsamples[count]=x;
   Ksamples[count]=y1;
   λsamples[count]=f(x,y1)[2][1]
   C0samples[count]=f(x,y1)[2][2]
   end
   end 
   
   #Define small number on the scale of the parameter λ
   ϵ=(βmax-βmin)/10^5
   count=0
   while count < N
   y=rand(Uniform(Kmin,Kmax))
   x0=rand(Uniform(βmin,βmax))
   x1=rand(Uniform(βmin,βmax))
   #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #horizontal line separating the two points    
   if g(x0,y)*g(x1,y) < 0 
   count+=1
   println(count)
   
   while abs(x1-x0) > ϵ && x1 < βmax && x1 > βmin
       x2=(x1+x0)/2;
           if g(x0,y)*g(x2,y) < 0 
           x1=x2
           else
           x0=x2
           end
       
       
       end
   
   
       βsamples[N+count]=x1;
       Ksamples[N+count]=y;
       λsamples[N+count]=f(x1,y)[2][1]
       C0samples[N+count]=f(x1,y)[2][2]
       end
       end 
   #Plot the MLE and the 2N points identified on the boundary
   aa4=scatter([βmle],[Kmle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="β",ylabel="K",label=false)
   display(aa4)
   for i in 1:2*N
   aa4=scatter!([βsamples[i]],[Ksamples[i]],markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
   end
   display(aa4)
   savefig(aa4,"BivariateβK.pdf")
   #Solve the model using the parameter values on the boundary of the bivariate profile
   Ctrace4 = zeros(length(tt),2*N)
   CU4=zeros(length(tt))
   CL4=zeros(length(tt))
   for i in 1:2*N
   Ctrace4[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
   end
       
   #Calculate the maximum/minimum envelope of the solutions    
   for i in 1:length(tt)
   CU4[i] = maximum(Ctrace4[i,:])
   CL4[i] = minimum(Ctrace4[i,:])
   end
       
   
   #Plot the family of solutions, the maximum/minimum envelope and the MLE
   q4=plot(tt,CL4,lw=0,fillrange=CU4,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
   q4=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
   savefig(q4,"BivariatepredictionsβK.pdf")


#5 bivriate(β,C0)    
function bivariateβC0(β,C0)
    function funβC0(a)
       return loglhood(data,[a[1],β,a[2],C0])
       end
       θG = [λmle,Kmle]
       lb=[λmin,Kmin]
       ub=[λmax,Kmax]
       (xopt,fopt)  = Optimise(funβC0,θG,lb,ub)
   llb=fopt-fmle
   return llb,xopt
   end 
   f(x,y) = bivariateβC0(x,y)
   g(x,y)=f(x,y)[1]-llstar
   
   #Define small parameter on the scale of parameter C0
   ϵ=(C0max-C0min)/10^4
   N=250
   λsamples=zeros(2*N)
   βsamples=zeros(2*N)
   Ksamples=zeros(2*N)
   C0samples=zeros(2*N)
   count=0
   
   #Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
   while count < N
   x=rand(Uniform(βmin,βmax))
   y0=rand(Uniform(C0min,C0max))
   y1=rand(Uniform(C0min,C0max))
   #If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #vertical line separating the two points
   if g(x,y0)*g(x,y1) < 0 
   count+=1
   println(count)
   while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
   y2=(y1+y0)/2;
       if g(x,y0)*g(x,y2) < 0 
       y1=y2
       else
       y0=y2
       end
   
   
   end
   
   βsamples[count]=x;
   C0samples[count]=y1;
   λsamples[count]=f(x,y1)[2][1]
   Ksamples[count]=f(x,y1)[2][2]
   end
   end 
   
   #Define small number on the scale of the parameter λ
   ϵ=(βmax-βmin)/10^5
   count=0
   while count < N
   y=rand(Uniform(C0min,C0max))
   x0=rand(Uniform(βmin,βmax))
   x1=rand(Uniform(βmin,βmax))
   #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #horizontal line separating the two points    
   if g(x0,y)*g(x1,y) < 0 
   count+=1
   println(count)
   
   while abs(x1-x0) > ϵ && x1 < βmax && x1 > βmin
       x2=(x1+x0)/2;
           if g(x0,y)*g(x2,y) < 0 
           x1=x2
           else
           x0=x2
           end
       
       
       end
   
   
       βsamples[N+count]=x1;
       C0samples[N+count]=y;
       λsamples[N+count]=f(x1,y)[2][1]
       Ksamples[N+count]=f(x1,y)[2][2]
       end
       end 
   #Plot the MLE and the 2N points identified on the boundary
   aa5=scatter([βmle],[C0mle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="C(0)",label=false)
   display(aa5)
   for i in 1:2*N
   aa5=scatter!([βsamples[i]],[C0samples[i]],markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
   end
   display(aa5)
   savefig(aa5,"BivariateβC0.pdf")
   #Solve the model using the parameter values on the boundary of the bivariate profile
   Ctrace5 = zeros(length(tt),2*N)
   CU5=zeros(length(tt))
   CL5=zeros(length(tt))
   for i in 1:2*N
   Ctrace5[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
   end
       
   #Calculate the maximum/minimum envelope of the solutions    
   for i in 1:length(tt)
   CU5[i] = maximum(Ctrace5[i,:])
   CL5[i] = minimum(Ctrace5[i,:])
   end
       
   
   #Plot the family of solutions, the maximum/minimum envelope and the MLE
   q5=plot(tt,CL5,lw=0,fillrange=CU5,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
   q5=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
   savefig(q5,"BivariatepredictionsβC0.pdf")

#6 bivriate(K,C0)    
function bivariateKC0(K,C0)
    function funKC0(a)
       return loglhood(data,[a[1],a[2],K,C0])
       end
       θG = [λmle,βmle]
       lb=[λmin,βmin]
       ub=[λmax,βmax]
       (xopt,fopt)  = Optimise(funKC0,θG,lb,ub)
   llb=fopt-fmle
   return llb,xopt
   end 
   f(x,y) = bivariateKC0(x,y)
   g(x,y)=f(x,y)[1]-llstar
   
   
   #Define small parameter on the scale of parameter C0
   ϵ=(C0max-C0min)/10^4
   N=250
   λsamples=zeros(2*N)
   βsamples=zeros(2*N)
   Ksamples=zeros(2*N)
   C0samples=zeros(2*N)
   count=0
   
   #Identify N points on the boundary by fixing values of λ and picking pairs of values of β 
   while count < N
   x=rand(Uniform(Kmin,Kmax))
   y0=rand(Uniform(C0min,C0max))
   y1=rand(Uniform(C0min,C0max))
   #If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #vertical line separating the two points
   if g(x,y0)*g(x,y1) < 0 
   count+=1
   println(count)
   while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
   y2=(y1+y0)/2;
       if g(x,y0)*g(x,y2) < 0 
       y1=y2
       else
       y0=y2
       end
   
   
   end
   
   Ksamples[count]=x;
   C0samples[count]=y1;
   λsamples[count]=f(x,y1)[2][1]
   βsamples[count]=f(x,y1)[2][2]
   end
   end 
   
   #Define small number on the scale of the parameter λ
   ϵ=(Kmax-Kmin)/10^5
   count=0
   while count < N
   y=rand(Uniform(C0min,C0max))
   x0=rand(Uniform(Kmin,Kmax))
   x1=rand(Uniform(Kmin,Kmax))
   #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
   #horizontal line separating the two points    
   if g(x0,y)*g(x1,y) < 0 
   count+=1
   println(count)
   
   while abs(x1-x0) > ϵ && x1 < Kmax && x1 > Kmin
       x2=(x1+x0)/2;
           if g(x0,y)*g(x2,y) < 0 
           x1=x2
           else
           x0=x2
           end
       
       
       end
   
   
       Ksamples[N+count]=x1;
       C0samples[N+count]=y;
       λsamples[N+count]=f(x1,y)[2][1]
       βsamples[N+count]=f(x1,y)[2][2]
       end
       end 
   #Plot the MLE and the 2N points identified on the boundary
   aa6=scatter([Kmle],[C0mle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="C(0)",label=false)
   display(aa6)
   for i in 1:2*N
   aa6=scatter!([Ksamples[i]],[C0samples[i]],markersize=3,markershape=:circle,markercolor=:gold,msw=0, ms=5,label=false)
   end
   display(aa6)
   savefig(aa6,"BivariateKC0.pdf")
   #Solve the model using the parameter values on the boundary of the bivariate profile
   Ctrace6 = zeros(length(tt),2*N)
   CU6=zeros(length(tt))
   CL6=zeros(length(tt))
   for i in 1:2*N
   Ctrace6[:,i]=model(tt,[λsamples[i],βsamples[i],Ksamples[i],C0samples[i]])
   end
       
   #Calculate the maximum/minimum envelope of the solutions    
   for i in 1:length(tt)
   CU6[i] = maximum(Ctrace6[i,:])
   CL6[i] = minimum(Ctrace6[i,:])
   end
       
   
   #Plot the family of solutions, the maximum/minimum envelope and the MLE
   q6=plot(tt,CL6,lw=0,fillrange=CU6,fillalpha=0.40,color=:purple,label=false,xlims=(0,maximum(tt)))
   q6=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=1,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
   savefig(q6,"BivariatepredictionsKC0.pdf")



# Compute the union of the three pair-wise profile predictions using the grid
CU_union=zeros(length(tt))
CL_union=zeros(length(tt))
for i in 1:length(tt)
CU_union[i]=max(CU1[i],CU2[i],CU3[i],CU4[i],CU5[i],CU6[i])
CL_union[i]=min(CL1[i],CL2[i],CL3[i],CL4[i],CL5[i],CL6[i])
end





#Plot the family of predictions made using the grid, the MLE and the prediction intervals defined by the full log-liklihood and the union of the three bivariate profile likelihood 
pp1=plot(tt,CL_union,lw=0,fillrange=CU_union,fillalpha=0.40,color=:gold,label=false,xlims=(0,maximum(tt)))
pp1=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
pp1=scatter!(t,data,legend=false,msw=0,ms=5,color=:darkorange,msa=:darkorange)

pp1=plot(tt,CL_union,lw=0,fillrange=CU_union,fillalpha=0.40,color=:gold,label=false,xlims=(0,maximum(tt)))




q1=plot(tt,CLF,lw=0,fillrange=CUF,fillalpha=0.40,color=:gold,label=false,xlims=(0,maximum(tt)))
q1=plot!(ymle,minimum(t),maximum(t),color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,4100),ylims=(0,100),xticks=[0,1000,2000,3000,4000],yticks=[0,25,50,75])
q1=plot!(tt,CL_union,lw=3,color=:purple,ls=:dash,legend=false)
q1=plot!(tt,CU_union,lw=3,color=:purple,ls=:dash,legend=false)
savefig(q1,"PredictionComparison.pdf")


df=1
llstar=-quantile(Chisq(df),0.95)/2
#Function to define univariate profile for λ    
function univariateλ(λ)
a=zeros(3)    
function funλ(a)
return loglhood(data,[λ,a[1],a[2],a[3]])
end
θG=[βmle,Kmle,C0mle]
lb=[βmin,Kmin,C0min] 
ub=[βmax,Kmax,C0max] 
(xopt,fopt)=Optimise(funλ,θG,lb,ub)
return fopt,xopt
end 
f(x) = univariateλ(x)[1]

#Take a grid of M points to plot the univariate profile likelihood
M=40;
λrange=LinRange(λmin,λmax,M)
ff=zeros(M)
for i in 1:M
    ff[i]=univariateλ(λrange[i])[1]
    println(i)
end

r1=plot(λrange,ff.-maximum(ff),ylims=(-3,0.1),legend=false,lw=3,color=:blue)
r1=hline!([llstar],lw=3)
r1=vline!([λmle],legend=false,xlabel="λ",ylabel="ll",lw=3,color=:green)

function univariateβ(β)
    a=zeros(3)    
    function funβ(a)
    return loglhood(data,[a[1],β,a[2],a[3]])
    end
    θG=[λmle,Kmle,C0mle]
    lb=[λmin,Kmin,C0min] 
    ub=[λmax,Kmax,C0max] 
    (xopt,fopt)=Optimise(funβ,θG,lb,ub)
    return fopt,xopt
    end 
    f(x) = univariateβ(x)[1]
    
    #Take a grid of M points to plot the univariate profile likelihood
    M=40;
    βrange=LinRange(βmin,βmax,M)
    ff=zeros(M)
    for i in 1:M
        ff[i]=univariateβ(βrange[i])[1]
        println(i)
    end
    
    r2=plot(βrange,ff.-maximum(ff),ylims=(-3,0.1),legend=false,lw=3,color=:blue)
    r2=hline!([llstar],lw=3)
    r2=vline!([βmle],legend=false,xlabel="β",ylabel="ll",lw=3,color=:green)


    
function univariateK(K)
    a=zeros(3)    
    function funK(a)
    return loglhood(data,[a[1],a[2],K,a[3]])
    end
    θG=[λmle,βmle,C0mle]
    lb=[λmin,βmin,C0min] 
    ub=[λmax,βmax,C0max] 
    (xopt,fopt)=Optimise(funK,θG,lb,ub)
    return fopt,xopt
    end 
    f(x) = univariateK(x)[1]
    
    #Take a grid of M points to plot the univariate profile likelihood
    M=40;
    Krange=LinRange(Kmin,Kmax,M)
    ff=zeros(M)
    for i in 1:M
        ff[i]=univariateK(Krange[i])[1]
        println(i)
    end
    
    r3=plot(Krange,ff.-maximum(ff),ylims=(-3,0.1),legend=false,lw=3,color=:blue)
    r3=hline!([llstar],lw=3)
    r3=vline!([Kmle],legend=false,xlabel="K",ylabel="ll",lw=3,color=:green)


    
function univariateC0(C0)
    a=zeros(3)    
    function funC0(a)
    return loglhood(data,[a[1],a[2],a[3],C0])
    end
    θG=[λmle,βmle,Kmle]
    lb=[λmin,βmin,Kmin] 
    ub=[λmax,βmax,Kmax] 
    (xopt,fopt)=Optimise(funC0,θG,lb,ub)
    return fopt,xopt
    end 
    f(x) = univariateC0(x)[1]
    
    #Take a grid of M points to plot the univariate profile likelihood
    M=40;
    C0range=LinRange(C0min,C0max,M)
    ff=zeros(M)
    for i in 1:M
        ff[i]=univariateC0(C0range[i])[1]
        println(i)
    end
    
    r4=plot(C0range,ff.-maximum(ff),ylims=(-3,0.1),legend=false,lw=3,color=:blue)
    r4=hline!([llstar],lw=3)
    r4=vline!([C0mle],legend=false,xlabel="C(0)",ylabel="ll",lw=3,color=:green)


r5=plot(r1,r2,r3,r4,layout=(4,1))
savefig(r5, "Univariate.pdf")