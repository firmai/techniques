/* Do file to import UCLA-LoPucki Bankruptcy Research Database
	This file imports the data from a .csv file and labels the variables. */

/* Change this path to point to the location of the data file */
cd "R:\Bankruptcy Research Database\Do File Design"
clear all
set memory 50m	
insheet using "UCLA-LoPucki Bankruptcy Research Database.csv", comma case clear

/* Drop cases that are Chapter 7 at filing */
drop if Disposition == "Chapter 7 at filing"

/* Convert dates to Stata date format.  
	'capture'  commands compensate for data imported directly from .xlsx files.  */
foreach v of var Date10k1Before-DateTurnEngage {
	local new= lower("`v'")
	capture gen `new'=date(`v', "MDY")
	capture format `new' %tdCCYY.NN.DD
	}



/* Identify firms that emerged from bankruptcy */
gen emerge = .
replace emerge = 1 if Emerge=="yes"
replace emerge = 0 if Emerge=="no"

/* Identify firms that filed a 10k after emerging */
gen emerg10k = .
replace emerg10k = 1 if Emerg10k=="yes" | Emerg10k=="yes (shell)" | Emerg10k=="20-F"
replace emerg10k = 0 if Emerg10k=="no"

/* Identify firms that adopted fresh start accounting after emerging */
gen freshstartaccounting = .
replace freshstartaccounting = 1 if FreshStartAccounting=="yes"
replace freshstartaccounting = 0 if FreshStartAccounting=="no" | FreshStartAccounting=="no informastion in 10-K"

/* Identify firms that refiled ever */
gen refile = 0
replace refile = 1 if Refile=="Refiled" | Refile=="Partial"

/* Identify firms that refiled within 5 years after confirmation.  Also see "wgtrefile5" */
gen refile5 = .
replace refile5 = 1 if Refile5=="emerged refile 5" | Refile5=="emerged part refile 5"
replace refile5 = 0 if Refile5=="emerged no refile 5"

/* Identify firms that refiled within the date range for which the BRD
	has representative data*/
gen wgtrefile = 0
replace wgtrefile = 1 if emerge == 1 & dateemerging < td(01jan2008)
replace wgtrefile = 1 if emerge == 1 & dateemerging == .

/* This is a binary variable of refile5 that includes only cases within a date range that 
	are representative of those at risk of refiling. */
gen refile5reprange = .
replace refile5reprange = 1 if wgtrefile == 1 & Refile5=="emerged refile 5" | Refile5=="emerged part refile 5"
replace refile5reprange = 0 if wgtrefile == 1 & Refile5=="emerged no refile 5"

/* Idenfity firms that indicated at filing an intention to sell all or substantially all assets. */
gen saleintended = .
replace saleintended = 1 if SaleIntended == "yes" | SaleIntended == "yes, buyer" |  SaleIntended == "yes, no buyer" | SaleIntended == "yes, contract" | SaleIntended == "yes, piecemeal"
replace saleintended = 0 if SaleIntended=="no"

/* Create a dummy variable for 363 Sale cases */
gen sale363 = Sale363=="yes"

/* Identify firms for which a Chapter 11 trustee was appointed before disposition */ 
gen trustee = .
replace trustee = 0 if Trustee=="no" | Trustee == "no hits" | Trustee=="denied" | Trustee=="partial, denied" | Trustee=="partial" | Trustee=="partial, denied re all"
replace trustee = 1 if Trustee=="yes"

/* Identify cases in which a request was made for appointment of a Chapter 11 trustee before disposition*/
gen trusteerequest = .
replace trusteerequest=0 if TrusteeRequest=="no"
replace trusteerequest=1 if TrusteeRequest=="yes"

/*This field indicates whether dispositions were by confirmation, conversion to chapter 7, or dismissal*/
gen disposedcasereprange = .
replace disposedcasereprange = 1 if Disposition =="confirmed" | Disposition =="confirmed converted" | Disposition =="confirmed dismissed" 
replace disposedcasereprange = 0 if Disposition =="converted" | Disposition =="dismissed"



/* Binary variables for prepackaged and prenegotiated cases */
gen prepack = 0
replace prepack = 1 if Prepackaged=="prepackaged"
gen preneg = 0
replace preneg = 1 if Prepackaged=="prenegotiated"
gen prepackpreneg = 0
replace prepackpreneg = 1 if Prepackaged=="prepackaged" | Prepackaged=="prenegotiated"

/* Identify whether the case had a fee reviewer or committee */
gen feereviewer = .
replace feereviewer = 1 if FeeReviewer=="fee auditor" | FeeReviewer=="fee committee" | FeeReviewer=="fee examiner" | FeeReviewer=="fee review committee" | FeeReviewer=="joint fee review committee" 
replace feereviewer = 0 if FeeReviewer=="no" | FeeReviewer=="no (denied)"

/* Create dummy variables related to court location */
gen shop = 0
replace shop = 1 if Shop=="Yes"
gen de = 0
replace de = 1 if DENYOther=="DE"
gen ny = 0
replace ny = 1 if DENYOther=="NY"
gen ot = 0
replace ot = 1 if DENYOther=="OT"

/* Create a dummy for cases that emerged, and whether they emerged after a whole or partial 363 sale */
gen emerge363sale = .
replace emerge363sale = 0 if emerge==1
replace emerge363sale = 1 if emerge==1 & sale363==1

lab def emerge363sale 0"Confirmed Emerge" 1"363 sale Emerge"
lab val emerge363sale emergetype
lab var shop "Filed away from headquarters"
lab var de "Filed in Delaware"
lab var ny "Filed in Manhattan Division Sdny"
lab var sale363 "Sale before confirmation"
lab def refile5 0"No Refile" 1"Refiled"
lab val refile5 refile5
lab var refile5 "Refiled within 5 years"
lab var feereviewer "Fee reviewer in case"

/*Start the LML code*/

/* Create dummy variables for judges */
gen walsh = 0
replace walsh = 1 if JudgeDisposition=="Peter J. Walsh"
gen walrath = 0
replace walrath = 1 if JudgeDisposition=="Mary F. Walrath"
gen lifland = 0
replace lifland = 1 if JudgeDisposition=="Burton R. Lifland"
gen balick = 0
replace balick = 1 if JudgeDisposition=="Helen S. Balick"
gen carey = 0
replace carey = 1 if JudgeDisposition=="Kevin J. Carey" 
gen gropper = 0
replace gropper = 1 if JudgeDisposition=="Allan L. Gropper"
gen beatty = 0
replace beatty = 1 if JudgeDisposition=="Prudence Carter Beatty"
gen robinson = 0
replace robinson = 1 if JudgeDisposition=="Sue L. Robinson"
gen gross = 0
replace gross = 1 if JudgeDisposition=="Kevin Gross"
gen sontchi = 0
replace sontchi = 1 if JudgeDisposition=="Christopher S. Sontchi"
gen drain = 0
replace drain = 1 if JudgeDisposition=="Robert D. Drain"

/* Create dummy variables for individual DIP Lead BK Atty firms */
gen weilgotshal = 0
replace weilgotshal = 1 if DipAtty=="Weil Gotshal"
gen skaddenarps = 0
replace skaddenarps = 1 if DipAtty=="Skadden Arps"
gen kirklandellis = 0
replace kirklandellis = 1 if DipAtty=="Kirkland Ellis"
gen willkiefarr = 0
replace willkiefar = 1 if DipAtty=="Willkie Farr"
gen jonesday = 0
replace jonesday = 1 if DipAtty=="Jones Day"
gen stutmantriester = 0
replace stutmantriester = 1 if DipAtty=="Stutman Triester"
gen lathamwatkins = 0
replace lathamwatkins = 1 if DipAtty=="Latham Watkins"

/* Create dummy variables for industry at the Division level */
gen mining = 0
replace mining = 1 if SICDivision=="B: Mining"
gen construction = 0
replace construction = 1 if SICDivision=="C: Construction"
gen manufacturing = 0
replace manufacturing = 1 if SICDivision=="D: Manufacturing"
gen transportation = 0
replace transportation = 1 if SICDivision=="E: Transportation, Communications, Electric, Gas"
gen wholesale = 0
replace wholesale = 1 if SICDivision=="F: WholesaleTrade"
gen retail = 0
replace retail = 1 if SICDivision=="G: Retail Trade"
gen finance = 0
replace finance = 1 if SICDivision=="H: Finance, Insurance, And Real Estate"
gen services = 0
replace services = 1 if SICDivision=="I: Services"

/* Create dummy variables for industry at the Major Group level */
gen oilgas = 0
replace oilgas = 1 if SICMajGroup=="13 Oil And Gas Extraction"
gen mfgtextile = 0
replace mfgtextile = 1 if SICMajGroup=="22 Textile Mill Products"
gen mfgchemicals = 0
replace mfgchemicals = 1 if SICMajGroup =="28 Chemicals and Allied Products"
gen mfgprimarymetals = 0
replace mfgprimarymetals = 1 if SICMajGroup=="33 Primary Metal Industries"
gen mfgmachinery = 0
replace mfgmachinery = 1 if SICMajGroup=="35 Industrial and Commercial Machinery and Computer Equipment"
gen mfgelectronic = 0
replace mfgelectronic = 1 if SICMajGroup=="36 Electronic And Other Electrical Equipment And Components"
gen mfgtransportation = 0
replace mfgtransportation = 1 if SICMajGroup=="37 Transportation Equipment"
gen transportationbyair = 0
replace transportationbyair = 1 if SICMajGroup=="45 Transportation By Air"
gen communications = 0
replace communications = 1 if SICMajGroup=="48 Communications"
gen utilities = 0
replace utilities = 1 if SICMajGroup=="49 Electric, Gas, And Sanitary Services"
gen wholedurablegoods = 0
replace wholedurablegoods = 1 if SICMajGroup=="50 WholesaleTrade-durable Goods"
gen wholegeneral = 0
replace wholegeneral = 1 if SICMajGroup=="53 General Merchandise Stores"
gen banksavings = 0
replace banksavings = 1 if SICMajGroup=="60 Depository Institutions"
gen creditinstitutions = 0
replace creditinstitutions = 1 if SICMajGroup =="61 Non-depository Credit Institutions"
gen insurancecarriers = 0
replace insurancecarriers = 1 if SICMajGroup=="63 Insurance Carriers"
gen businessservices = 0
replace businessservices = 1 if SICMajGroup=="73 Business Services"
gen healthservices = 0
replace healthservices = 1 if SICMajGroup=="80 Health Services"

/* Create dummy variables for industry at the Industry Group level */

gen telecom = 0
replace telecom = 1 if SICIndustryGroup=="481 Telephone Communications"
gen mfgmotorvehicle = 0
replace mfgmotorvehicle = 1 if SICIndustryGroup=="371 Motor Vehicles and Equipment"
gen mfgsteelworks = 0
replace mfgsteelworks = 1 if SICIndustryGroup=="331 Steel Works, Blast Furnaces, And Rolling And"
gen crudepetroleum = 0
replace crudepetroleum = 1 if SICIndustryGroup=="131 Crude Petroleum and Natural Gas"

/* Identify cases in which the plan substantively consolidates debtor entities */
gen subcon = .
replace subcon = 1 if SubCon=="subcon"
replace subcon = 0 if SubCon=="no"

/*This field indicates the participation of a turnaround manager in the range since December 2003*/
gen turnreprange = .
replace turnreprange = 1 if TurnRepRange=="yes"
replace turnreprange = 0 if TurnRepRange=="no"

/* Weight variable for turnaround manager analysis */
gen wgtturnrep = 0
replace wgtturnrep = 1 if TurnRepRange!=""

/* Identify turnaround management firms */

gen alixpartners = 0
replace alixpartners = 1 if TurnFirm=="AlixPartners"
replace alixpartners = . if turnreprange == .

gen alvarezmarsal = 0
replace alvarezmarsal = 1 if TurnFirm=="Alvarez & Marsal"
replace alvarezmarsal = . if turnreprange == .

/* For cases in the representive range, identify whether turnaround management officer is the CEO */
gen turnceo = 0
replace turnceo = 1 if TurnOffice=="Chief Executive Officer"
replace turnceo = 1 if TurnOffice=="Interim Chief Executive Officer"
replace turnceo = 1 if TurnOffice=="President, Chief Restructuring Officer"
replace turnceo = . if turnreprange == .

/* For cases in the representive range, identify whether the turnaround manager was an officer of the firm */
gen turnofficer = .
replace turnofficer = 1 if TurnOffice=="Chief Executive Officer"
replace turnofficer = 1 if TurnOffice=="Interim Chief Executive Officer"
replace turnofficer = 1 if TurnOffice=="President, Chief Restructuring Officer"
replace turnofficer = 1 if TurnOffice=="Chief Financial Officer"
replace turnofficer = 1 if TurnOffice=="Chief Operating Officer"
replace turnofficer = 1 if TurnOffice=="Vice President Restructuring"
replace turnofficer = 0 if TurnOffice=="no office"
replace turnofficer = 0 if TurnOffice=="no turnfirm"
replace turnofficer = 0 if TurnOffice=="Chief Restructuring Advisor"
replace turnofficer = 0 if TurnOffice=="Crisis Manager"
replace turnofficer = . if turnreprange == .

/* Set the date for engaging a turnaround managment firm to missing if it is outside the 
	range of cases that are representative of the universe of cases */
replace dateturnengage = . if turnreprange == .

/* Identify Delaware incorporated firms */
gen deinc = 0
replace deinc = 1 if IncPublic=="DE"

/* Identify cases in which tort debt caused the bankruptcy filing */
gen tortcause = 1
replace tortcause = 0 if TortCause=="Not tort"

/* Identify cases with a narrower definition of "tort" */
gen tortcausenarrow = 1
replace tortcausenarrow = 0 if TortCause=="Not tort" | TortCause=="Fraud" | TortCause=="Fraud?" |  TortCause=="Pension" | TortCause=="Environmental"

/* Identify asbestos cases */
gen asbestos = 0
replace asbestos = 1 if TortCause=="Asbestos"

/* Identify fraud cases */
gen fraud = 0
replace fraud = 1 if TortCause=="Fraud" | TortCause=="Fraud?"

/* Identify cases in which bankruptcy was filed by the creditors */
gen involuntary = 0
replace involuntary = 1 if Voluntary=="involuntary" | Voluntary=="both"

gen commretirees = .
replace commretirees = 1 if CommRetirees=="yes"
replace commretirees = 0 if CommRetirees=="no" | CommRetirees=="no; withdrawn" | CommRetirees=="no; denied"

gen examiner = .
replace examiner = 1 if Examiner=="yes"
replace examiner = 0 if Examiner=="no"

gen examinerrequest = .
replace examinerrequest = 1 if ExaminerRequest=="yes"
replace examinerrequest = 0 if ExaminerRequest=="no"

gen intercompanyclaims = .
replace intercompanyclaims = 1 if InterCompanyClaims=="yes"
replace intercompanyclaims = 0 if InterCompanyClaims=="no"


/*Import numeric fields*/

gen bondpricemoveduring = BondPriceMoveDuring
gen bondpricefile = BondPriceFile

/* Set missing values for bond price data */
recode bondpricefile (999999 = .)
recode bondpricemoveduring (-999999/-999000 = .)

/* Set missing values for CEO end date */
replace dateceoend = . if dateceoend==td(09sep9999)

gen assetsbefore = AssetsBefore
gen assetscurrdollar = AssetsCurrDollar
gen assetsemerging = AssetsEmerging
gen assetspetcurrdollar = AssetsPetCurrDollar
gen assetspetition = AssetsPetition
gen bondpricedisp = BondPriceDisp
gen ceodaysbefore = datefiled - dateceobegin
gen ceodaysafter = dateceoend - datefiled
gen claimssecdisclostate = ClaimsSecDiscloState
gen claimsunsec = ClaimsUnsec
gen cpindexatconf = CPIndexAtConf
gen cpindexatfiling = CPIndexAtFiling
gen daysfiledto363 = DaysFiledTo363
gen daysemergetorefile = DaysEmergeToRefile
gen distribequity = DistribEquity
gen distribsecdisclostate = DistribSecDiscloState
gen distribunsec = DistribUnsec
gen dktnumconf = DktNumConf
gen ebitbefore = EbitBefore
gen ebitdabefore = EbitdaBefore
gen ebitdaemerging = EbitdaEmerging
gen ebitemerging = EbitEmerging
gen emplemerging = EmplEmerging
gen emplunionbefore = EmplUnionBefore
gen emplunionemerging = EmplUnionEmerging
gen gdpdisp = GdpDisp
gen gdpfiling = GdpFiling
gen gdprefiling = GdpRefiling
gen gdpyear1aftdisp = GdpYear1AftDisp
gen gdpyear1beffile = GdpYear1BefFile
gen gdpyear2aftdisp = GdpYear2AftDisp
gen gdpyear2beffile = GdpYear2BefFile
gen headcourtcitytode = HeadCourtCityToDE
gen hqtoforum = HqToForum
gen hqtohqctcity = HqToHqCtCity
gen incomebebefore = IncomeBEBefore
gen incomebeemerging = IncomeBEEmerging
gen intercompanypct = InterCompanyPct
gen liabbefore = LiabBefore
gen liabcurrdollar = LiabCurrDollar
gen liabemerging = LiabEmerging
gen liabpetcurrdollar = LiabPetCurrDollar
gen liabpetition = LiabPetition
gen netincomebefore = NetIncomeBefore
gen netincomeemerging = NetIncomeEmerging
gen numberfiling = NumberFiling
gen prime1yearaftdisp = Prime1YearAftDisp
gen prime1yearbeffile = Prime1YearBefFile
gen prime2yearaftdisp = Prime2YearAftDisp
gen prime2yearbeffile = Prime2YearBefFile
gen primedisp = PrimeDisp
gen primefiling = PrimeFiling
gen primerefiling = PrimeRefiling
gen proffees10k = ProfFees10k
gen salesbefore = SalesBefore
gen salescurrdollar = SalesCurrDollar
gen salesemerging = SalesEmerging
gen daysin = DaysIn
gen emplbefore = EmplBefore
gen yeardisposed = YearDisposed
gen yearfiled = YearFiled
gen yearconfirmed = YearConfirmed
gen yearsemergetorefile = YearsEmergeToRefile

/* Calculate the percentage distribution to unsecured creditors. */
gen distribunsecpct = distribunsec / claimsunsec

/*Log numeric fields*/

gen lnassetsbefore = log(AssetsBefore)
gen lnassetscurrdollar = log(AssetsCurrDollar)
gen lnassetsemerging = log(AssetsEmerging)
gen lnassetspetcurrdollar = log(AssetsPetCurrDollar)
gen lnassetspetition = log(AssetsPetition)
gen lnbondpricedisp = log(BondPriceDisp)
gen lnbondpricefile = log(BondPriceFile)
gen lnceodaysbefore = log(ceodaysbefore)
gen lnceodaysaftter = log(ceodaysafter)
gen lnclaimssecdisclostate = log(ClaimsSecDiscloState +0.001)
gen lnclaimsunsec = log(ClaimsUnsec)
gen lndaysfiledto363 = log(DaysFiledTo363)
gen lndaysemergetorefile = log(DaysEmergeToRefile)
gen lndistribsecdisclostate = log(DistribSecDiscloState +0.001)
gen lndistribunsec = log(DistribUnsec +0.001)
gen lndktnumconf = log(DktNumConf)
gen lnemplemerging = log(EmplEmerging)
gen lnemplunionbefore1 = log(EmplUnionBefore +1)
gen lnemplunionemerging1 = log(EmplUnionEmerging +1)
gen lngdpdisp = log(GdpDisp)
gen lngdpfiling = log(GdpFiling)
gen lngdprefiling = log(GdpRefiling)
gen lngdpyear1aftdisp = log(GdpYear1AftDisp)
gen lngdpyear1beffile = log(GdpYear1BefFile)
gen lngdpyear2aftdisp = log(GdpYear2AftDisp)
gen lngdpyear2beffile = log(GdpYear2BefFile)
gen lnheadcourtcitytode = log(HeadCourtCityToDE)
gen lnhqtoforum = log(HqToForum)
gen lnhqtohqctcity = log(HqToHqCtCity)
gen lnliabbefore = log(LiabBefore)
gen lnliabcurrdollar = log(LiabCurrDollar)
gen lnliabemerging = log(LiabEmerging)
gen lnliabpetcurrdollar = log(LiabPetCurrDollar)
gen lnliabpetition = log(LiabPetition)
gen lnnumberfiling = log(NumberFiling)
gen lnproffees10k = log(ProfFees10k)
gen lnsalesbefore = log(SalesBefore)
gen lnsalescurrdollar = log(SalesCurrDollar)
gen lnsalesemerging = log(SalesEmerging +0.001)
gen lndaysin = log(DaysIn)
gen lnemplbefore = log(EmplBefore)
gen lnyearsemergetorefile = log(YearsEmergeToRefile)

/*Calculated fields*/

gen lnchangeassets = lnassetsemerging - lnassetsbefore
gen lnchangebondprice = lnbondpricedisp - lnbondpricefile
gen lnchangeempl = lnemplemerging - lnemplbefore
gen lnchangeunionempl1 = lnemplunionemerging1 - lnemplunionbefore1
gen lnchangegdp = lngdpdisp - lngdpfiling
gen lnchangegdp1beftofiling = lngdpfiling - lngdpyear1beffile
gen lnchangegdp2beftofiling = lngdpfiling - lngdpyear2beffile
gen lnchangegdp2befto1bef = lngdpyear1beffile - lngdpyear2beffile
gen lnchangegdpdispto1aft = lngdpyear1aftdisp - lngdpdisp
gen lnchangegdpdispto2aft = lngdpyear2aftdisp - lngdpdisp
gen lnchangegdp1aftto2aft = lngdpyear2aftdisp - lngdpyear1aftdisp
gen lnchangeliab = lnliabemerging - lnliabbefore
gen lnchangeliabpetition = lnliabemerging - lnliabpetition
gen lnchangesales = lnsalesemerging - lnsalesbefore
gen changeebit = ebitemerging/assetsemerging - ebitbefore/assetsbefore
gen changeebitda = ebitdaemerging/assetsemerging -  ebitdabefore/assetsbefore
gen changeincomebee = IncomeBEEmerging/assetsemerging - IncomeBEBefore/assetsbefore
gen changenetincome = NetIncomeEmerging/assetsemerging - NetIncomeBefore/assetsbefore
gen changeprime	= primedisp - primefiling
gen changeprime1beftofile	= primefiling - prime1yearbeffile
gen changeprime2beftofile = primefiling - prime2yearbeffile
gen changeprime2befto1bef = prime1yearbeffile - prime2yearbeffile
gen changeprimedispto1aft = prime1yearaftdisp - primedisp
gen changeprimedispto2aft = prime2yearaftdisp - primedisp
gen changeprime1aftto2aft = prime2yearaftdisp - prime1yearaftdisp

gen solvencybefore = 0
replace solvencybefore = . if liabbefore == .
replace solvencybefore = 1 if assetsbefore >= liabbefore
gen solvencyemerging = 0
replace solvencyemerging = 1 if assetsemerging > liabemerging
replace solvencyemerging = . if liabemerging == .
gen solvencypetition = 0
replace solvencypetition = 1 if assetspetition > liabpetition
replace solvencypetition = . if liabpetition == .
replace solvencypetition = . if assetspetition == .

gen equitybefore = (assetsbefore - liabbefore)/assetsbefore
gen equityemerging = (assetsemerging - liabemerging)/assetsemerging
gen equitypetition = (assetspetition - liabpetition)/assetspetition
gen changeequity = equityemerging - equitybefore
gen lnchangeequity2 = (lnassetsemerging-lnliabemerging) - (lnassetsbefore-lnliabbefore)

gen bondpricerise = 0
replace bondpricerise = 1 if bondpricedisp > bondpricefile
replace bondpricerise = . if bondpricedisp == .
replace bondpricerise = . if bondpricefile == .

gen ceoturnoverduring = 0
replace ceoturnoverduring = 1 if dateceoend < datedisp
replace ceoturnoverduring = . if dateceoend ==.
replace ceoturnoverduring = . if datedisp ==.

gen ceoturnoverduring90 = 0
replace ceoturnoverduring90 = 1 if dateceoend < (datedisp + 90)
replace ceoturnoverduring90 = . if dateceoend ==.
replace ceoturnoverduring90 = . if datedisp ==.

gen daysconftoeffective = dateeffective - dateconfirmed
gen daysfiledtoemerging = dateemerging - datefiled

/* run "BRD Labels.do"*/

/*lab val emerge10k emerge
lab var emerge10k "Emerged company filed form 10-K within three years after confirmation"*/

/*lab val freshstartaccounting freshstartaccounting
lab var freshstartaccounting "Emerging company adopted fresh start accounting"*/

/*lab def refile5 0 "No Refile" 1"Refile"
lab val refile5 refile5
lab var refile5 "Company emerged from bankruptcy and refiled within five year of confirmation"*/

/*lab def saleintended 0"No" 1"Yes" 2"No Data"
lab val saleintended saleintended
lab var saleintended  "Company intended at time of case filing to sell its business"*/

/*lab val trustee trustee
lab var trustee "Chapter 11 trustee was appointed prior to disposition"*/

/*lab def trusteerequest 0"No" 1"Yes" 
lab val trusteerequest trusteerequest
lab var trusteerequest "Name of the party requesting Chapter 11 trustee appointment"*/

/*lab var lnassetsbef "Log of assets before filing"
lab var lnassetsemerging "Log of assets after emerging"
lab var lnemplbef "Log of employees before filing"
lab var lnsalesbef "Log of sales before filing"
lab var lnliabbef "Log of liabilities before filing"
lab var lnliabpet "Log of liabilities at filing"
lab var lndaysin "Log of days in bankruptcy"*/


/*  Start the Regression Engine! */

/*logit refiler5 DE yearconfirmed PrimeDisp, robust
logit refiler5 i.DE##c.yearconfirmed PrimeDisp, robust
logit refiler5 i.DE##c.yearconfirmed primedisp ebitemerging, robust
logit refiler5 DE primedisp ebitemerging, robust
logit refiler5 DE yearconfirmed PrimeDisp if e(sample), robust*/

logit refile5reprange de yearconfirmed PrimeDisp, robust
logit refile5reprange i.de##c.yearconfirmed PrimeDisp, robust
logit refile5reprange i.de##c.yearconfirmed primedisp ebitemerging, robust
logit refile5reprange de primedisp ebitemerging, robust
logit refile5reprange de yearconfirmed PrimeDisp if e(sample), robust
logit refile5reprange primedisp ebitemerging lnemplemerging solvencyemerging, robust



