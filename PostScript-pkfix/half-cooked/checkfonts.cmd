@echo off
setlocal
for %%f in (cmss9 cmmib9 eufm6 ecrb0900 ecti0900 ecrb1728 eufm5 cmss8 cmssbx10 cmmib7 cmmib10 cmsy8 cmsy10 cmr8 cmex7 cmsy6 eufm9 cmr6 msam10 cmr9 cmex9 cmmi6 line10 cmsy9 cmmi9 eufm7 cmbx10 msbm7 cmr5 eufm10 ecrm0800 ecrm0600 ecrm0700 msbm5 lcircle10 msbm10 cmmi5 cmsy5 cmss10 msam6 cmex10 ecti1000 eccc1000 ecrb1200 cmr10 cmr7 cmmi10 cmmi7 ecrm0900 cmsy7 ecrb1000 ecrm1000 ecrb2488 ecrb1440) do (
  echo ==== %%f ====
  kpsewhich %%f.tfm
  kpsewhich %%f.pfb
)
endlocal