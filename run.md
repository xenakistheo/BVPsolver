

Below there are seven different problems, three different models, and seven different training configurations. I want to run all 7*3*7 = 147 configurations.

Moreover, I want to run each configuration five times with different seeds.

What is the best way to launch all these jobs in on a cluster?
Perhaps I should check how many jobs can be run in parallel and then launch them in batches. Job array? 

seeds = [11, 12, 13, 14, 15]

--problem 
[pollu, rober, vanderpol, hires, orego, davidskodje, brusselator, ]

--model 
[stiff, mlp, GELU-scaled]


[
--pretraining none --training shooting
--pretraining derivmatch --training shooting   
--pretraining none --training shapovalova      
--pretraining derivmatch --training shapovalova
--pretraining derivmatch --training none
--pretraining none --training collocation 
--pretraining derivmatch --training collocation
]

sacctmgr show assoc user=$USER format=Account,User,MaxJobs,MaxSubmitJobs,QOS -p
Account|User|MaxJobs|MaxSubmit|QOS|
mit_general|txenakis||500|normal|
[txenakis@login007 ~]$ sshare -U -u $USER
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare 
-------------------- ---------- ---------- ----------- ----------- ------------- ---------- 
mit_general            txenakis          1    0.000105       22701      0.000001   0.151268 
[txenakis@login007 ~]$ scontrol show partition mit_normal
PartitionName=mit_normal
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=YES QoS=mit_normal
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO ExclusiveTopo=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=12:00:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=node[1600-1619,1622-1625,2704-2705,3103-3114,3303-3314]
   PriorityJobFactor=1000 PriorityTier=25 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=5120 TotalNodes=50 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED
   TRES=cpu=5120,mem=23515231M,node=50,billing=5120




   ## Supercloud
   sacctmgr show assoc user=$USER format=Account,User,MaxJobs,MaxSubmitJobs,QOS -p
Account|User|MaxJobs|MaxSubmit|QOS|
default_group|txenakis|20|20|high,normal|
default_group|txenakis|20|20|high,normal|
default_group|txenakis|1|1|high,normal|
default_group|txenakis|1|1|high,normal|
default_group|txenakis|1|1|high,normal|
default_group|txenakis|240|240|high,normal|
default_group|txenakis|240|240|high,normal|
txenakis@login-2:~/projects/BVPsolver$ sshare -U -u $USER
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare 
-------------------- ---------- ---------- ----------- ----------- ------------- ---------- 
default_group          txenakis          1    0.000031        3342      0.000000   0.020084 
default_group          txenakis          1    0.000031           0      0.000000   0.999969 
default_group          txenakis          1    0.000031           0      0.000000   0.999969 
default_group          txenakis          1    0.000031      242993      0.000017   0.010850 
default_group          txenakis          1    0.000031           0      0.000000   0.999969 
default_group          txenakis          1    0.000031    19869495      0.001381   0.003451 
default_group          txenakis          1    0.000031           6      0.000000   0.032582 
txenakis@login-2:~/projects/BVPsolver$ scontrol show partition xeon-p8
PartitionName=xeon-p8
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=N/A
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=YES ExclusiveTopo=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=4-04:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=d-3-1-[3-4],d-3-2-[1-4],d-3-3-[1-4],d-3-4-[1-4],d-3-5-[1-4],d-3-6-[1-4],d-3-7-[1-4],d-3-8-[1-4],d-3-9-[1-4],d-3-10-[1-4],d-3-11-[1-4],d-3-12-[1-4],d-3-13-[1-4],d-3-14-[1-4],d-4-1-[1-4],d-4-2-[1-4],d-4-3-[1-4],d-4-4-[1-4],d-4-5-[1-4],d-4-6-[1-4],d-4-7-[1-4],d-4-8-[1-4],d-4-9-[1-4],d-4-10-[1-4],d-4-11-[1-4],d-4-12-[1-4],d-4-13-[1-4],d-4-14-[1-4],d-4-15-[1-4],d-5-4-[1-4],d-5-5-[1-4],d-5-6-[1-4],d-5-7-[1-4],d-5-8-[1-4],d-5-9-[1-4],d-5-10-[1-4],d-5-11-[1-4],d-5-12-[1-4],d-5-13-[1-4],d-5-14-[1-4],d-5-15-[1-4],d-6-1-[1-4],d-6-2-[1-4],d-6-3-[1-4],d-6-4-[1-4],d-6-5-[1-4],d-6-6-[1-4],d-6-7-[1-4],d-6-8-[1-4],d-6-9-[1-4],d-6-10-[1-4],d-6-11-[1-4],d-6-12-[1-4],d-6-13-[1-4],d-6-14-[1-4],d-6-15-[1-4],d-16-1-[1-4],d-16-2-[1-4],d-16-3-[1-4],d-16-4-[1-4],d-16-5-[1-4],d-16-6-[1-4],d-16-7-[1-4],d-16-8-[1-4],d-16-9-[1-4],d-16-10-[1-4],d-16-11-[1-4],d-16-12-[1-4],d-16-13-[1-4],d-16-14-[1-4],d-16-15-[1-4],d-17-1-[1-4],d-17-2-[1-4],d-17-3-[1-4],d-17-4-[1-4],d-17-5-[1-4],d-17-6-[1-4],d-17-7-[1-4],d-17-8-[1-4],d-17-9-[1-4],d-17-10-[1-4],d-17-11-[1-4],d-17-12-[1-4],d-17-13-[1-4],d-17-14-[1-4],d-17-15-[1-4],d-18-1-[1-4],d-18-2-[1-4],d-18-3-[1-4],d-18-4-[1-4],d-18-5-[1-4],d-18-6-[1-4],d-18-7-[1-4],d-18-8-[1-4],d-18-9-[1-4],d-18-10-[1-4],d-18-11-[1-4],d-18-12-[1-4],d-18-13-[1-4],d-18-14-[1-4],d-18-15-[1-4],d-19-1-[1-4],d-19-2-[1-4],d-19-3-[1-4],d-19-4-[1-4],d-19-5-[1-4],d-19-6-[1-4],d-19-7-[1-4],d-19-8-[1-4],d-19-9-[1-4],d-19-10-[1-4],d-19-11-[1-4],d-19-12-[1-4],d-19-13-[1-4],d-19-14-[1-4],d-19-15-[1-4]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=REQUEUE
   State=UP TotalCPUs=22176 TotalNodes=462 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=4000 MaxMemPerNode=UNLIMITED
   TRES=cpu=22176,mem=86625G,node=462,billing=22176

txenakis@login-2:~/projects/BVPsolver$ 
