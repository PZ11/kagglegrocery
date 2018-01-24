
CREATE TABLE pzhang.g_train (
id  INTEGER, 
salesdate DATE,
store_nbr SMALLINT,
item_nbr INTEGER,
unit_sales DECIMAL(18,3) ,
onpromotion byteint
);

CREATE MULTISET TABLE pzhang.g_test
     (
      id INTEGER,
      salesdate DATE FORMAT 'YYYY-MM-DD',
      store_nbr SMALLINT,
      item_nbr INTEGER,    
      onpromotion BYTEINT)
PRIMARY INDEX ( id );

############# Add life on test data 
CREATE MULTISET TABLE pzhang.g_test_2
     (
      id INTEGER,
      salesdate DATE FORMAT 'YYYY-MM-DD',
      store_nbr SMALLINT,
      item_nbr INTEGER,    
      onpromotion BYTEINT,
      life char(1)
      )
PRIMARY INDEX ( id );

insert into pzhang.g_test_2
sel id, salesdate, store_nbr, item_nbr, onpromotion, 'm' from pzhang.g_test 
;

update g
from  pzhang.g_test_2 g, 
( sel store_nbr, item_nbr from  pzhang.g_test 
where (store_nbr, item_nbr ) not in ( 
select store_nbr, item_nbr from pzhang.g_train group by 1,2 )
group by 1,2
) t
set life = 'n'
where g.store_nbr = t.store_nbr
and g.item_nbr = t.item_nbr
;

update g
from  pzhang.g_test_2 g, 
( select store_nbr, item_nbr, min(salesdate) as min_date
 from pzhang.g_train group by 1,2 
having min_date >= date'2016-08-01' 
) t
set life = 'p'
where g.store_nbr = t.store_nbr
and g.item_nbr = t.item_nbr
;

------------------------------------------------------------------------------
--- 44421 of total 210654 store/items in test are new 
sel store_nbr, item_nbr from  pzhang.g_test 
where (store_nbr, item_nbr ) not in ( 
select store_nbr, item_nbr from pzhang.g_train group by 1,2 )
group by 1,2 

--- 8472 items has data in train, but not in test. Likely due to discontinue . 
sel store_nbr, item_nbr, sum(unit_sales) from  pzhang.g_train 
where (store_nbr, item_nbr ) not in ( 
select store_nbr, item_nbr from pzhang.g_test group by 1,2 )
group by 1,2 

-- only 1382 new store/items after 20170701 
sel store_nbr, item_nbr  
from pzhang.g_train 
where salesdate > date'2017-07-01'  
and  (store_nbr, item_nbr ) not in ( 
select store_nbr, item_nbr from pzhang.g_train where salesdate <= date'2017-07-01' group by 1,2 )
group by 1,2 
;
-- 23516 store/items are new after 20160801
-- 151169 store/items are mature with start date before 20160801
select store_nbr, item_nbr, min(salesdate) as min_date
 from pzhang.g_train group by 1,2 
having min_date >= date'2016-08-01' 
;

############################################################
## Life 'O' means old SKU that doesn't exist in test 

CREATE MULTISET TABLE pzhang.g_train_2
     (
      id INTEGER,
      salesdate DATE FORMAT 'YYYY-MM-DD',
      store_nbr SMALLINT,
      item_nbr INTEGER,
      unit_sales DECIMAL(18,3),
      onpromotion BYTEINT,
      life char(1)
      )
PRIMARY INDEX ( id );

DEL FROM  pzhang.g_train_2 ; 
INSERT INTO  pzhang.g_train_2
sel id, salesdate, tr.store_nbr, tr.item_nbr, unit_sales,
	onpromotion ,
	COALESCE(ts.life,'o')
from pzhang.g_train tr
LEFT JOIN 
( SEL store_nbr, item_nbr, life from pzhang.g_test_2 GROUP  BY 1,2,3 ) ts
ON  tr.store_nbr = ts.store_nbr 
AND tr.item_nbr = ts.item_nbr 
;


--!! Find highly seasonal SKUs on Aug 
############################################################

CREATE MULTISET TABLE pzhang.g_test_fcst_t007
     (
      id INTEGER,
      fcst DECIMAL(15,4)
      )
PRIMARY INDEX ( id );

############################################################


DROP TABLE pzhang.g_full ; 

CREATE MULTISET TABLE pzhang.g_full
     (
      store_nbr SMALLINT,
      item_nbr INTEGER,
      salesdate DATE FORMAT 'YYYY-MM-DD',

      sales DECIMAL(18,3),
      fcst DECIMAL(18,3),
      onpromotion BYTEINT,
      life char(1)
      )
PRIMARY INDEX ( store_nbr, item_nbr );


INSERT INTO pzhang.g_full
SELECT 
store_nbr, item_nbr, salesdate, 
unit_sales as sales, 0 as fcst,
zeroifnull(onpromotion),  life
FROM 
 pzhang.g_train_2 tr
 ; INSERT INTO pzhang.g_full
SELECT  
store_nbr, item_nbr, salesdate,  0 as sales, fcst, 
zeroifnull(onpromotion), life  
from  
pzhang.g_test_fcst_t007 f
inner join pzhang.g_test_2 t
on f.id = t.id
;

------------------------------------------------------------------------
------------------------------------------------------------------------
------------------------------------------------------------------------
-- discontinued SKUs, no sales in last 3 month.
-- need set forecast to zero. 
	SELECT  f.* 
	FROM
	(	
	sel item_nbr, store_nbr, salesdate, fcst 
	from  pzhang.g_full   where salesdate > '2017-08-15' 
	) f
	INNER JOIN 
	(
		sel store_nbr, item_nbr, 
		min(salesdate) as all_min_date, 
		max(salesdate) as all_max_date,  
		 max(salesdate)  - min(salesdate) + 1   as salesdays, 
	 	sum(sales)  as all_sales,
	 	all_sales / salesdays as avg_sales, 
	 	average(sales) as avg_sales_nozerodays
		from  pzhang.g_full  where life <> 'n' and salesdate <='2017-08-15'
		having all_max_date <= '2017-05-15' 
		group by 1 ,2
	) t
	ON  f.store_nbr = t.store_nbr
	and f.item_nbr = t.item_nbr 
	;

-----------------------------------------------------
---------- Load Val fcst July 26- Aug 10 
create table  pzhang.g_val_fcst
(
store_nbr INTEGER,
      item_nbr INTEGER,
      level_3 date,
      unit_sales Decimal(15,4),
pred_sales Decimal(15,4), 
      salesdate date)
PRIMARY INDEX ( item_nbr );
	
update f
	from  pzhang.g_full f, pzhang.g_val_fcst v
	set fcst = v.pred_sales
	where f.store_nbr = v.store_nbr 
	and f.item_nbr  = v.item_nbr
	and f.salesdate = v.level_3
;

######################################################################## 
######################################################################## 
######################################################################## 
update  pzhang.g_full set sales  = 0  where sales < 0  ;
### After 0.03% under after LN conversion.
sel  sum(unit_sales) , sum(pred_sales), 
sum(pred_sales) / sum(unit_sales)  - 1 , 
sum(ln(unit_sales + 1 ) ), sum(ln(pred_sales +1)),
sum(ln(pred_sales +1))  / sum(ln(unit_sales + 1 ) ) - 1 
	from  pzhang.g_val_fcst v 
;

update  pzhang.g_full set sales  = 0  where sales < 0  ;

###### Calendar 
drop table pzhang.g_fiscalcalendar ; 
create table pzhang.g_fiscalcalendar as  macysperf_cpdb.fiscalcalendar  with data and stats;

del pzhang.g_fiscalcalendar where yearplusweekno < 201301 ; 
del pzhang.g_fiscalcalendar where yearplusweekno > 201801 ;


update pzhang.g_fiscalcalendar set startdate = startdate - 25 ; 
update pzhang.g_fiscalcalendar set enddate = enddate - 25 ; 


DROP TABLE pzhang.g_calendar ; 

CREATE TABLE  pzhang.g_calendar 
     (
      fiscalyear INTEGER NOT NULL,
      fiscalquarter INTEGER NOT NULL,
      fiscalperiod INTEGER NOT NULL,
      fiscalweek INTEGER NOT NULL,
      yearplusweekno INTEGER NOT NULL,
      daynumber INTEGER NOT NULL ,
      salesdate DATE FORMAT 'yyyymmdd' NOT NULL
      )
UNIQUE PRIMARY INDEX ( salesdate)
;

INSERT INTO   pzhang.g_calendar 
SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),1 as daynumber,startdate FROM pzhang.g_fiscalcalendar 	
UNION ALL SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),2 as daynumber,startdate + 1  FROM pzhang.g_fiscalcalendar 
UNION ALL SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),3 as daynumber,startdate + 2  FROM pzhang.g_fiscalcalendar 
UNION ALL SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),4 as daynumber,startdate + 3  FROM pzhang.g_fiscalcalendar      
UNION ALL  SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),5 as daynumber,startdate + 4  FROM pzhang.g_fiscalcalendar 
UNION ALL SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),6 as daynumber,startdate + 5   FROM pzhang.g_fiscalcalendar 
UNION ALL SELECT (fiscalyear),(fiscalquarter),(fiscalperiod),(fiscalweek),(yearplusweekno),7 as daynumber,startdate + 6   FROM pzhang.g_fiscalcalendar 
;

--------------- check sales-fcst on classes
-- need left join to fcst 
select s.family,  s.salesdate, fiscalyear, fiscalperiod, fiscalweek, yearplusweekno, daynumber,  
s.sales, f.fcst 
from 
	( sel family,  salesdate, sum( ln(sales + 1 )) as sales 
	from  pzhang.g_full f, g_item   i
		where f.item_nbr = i.item_nbr
	group by 1 ,2 ) s
inner join pzhang.g_calendar c
on s.salesdate = c.salesdate
left join 
(

	sel family,  salesdate, sum( ln(pred_sales + 1 )) as fcst 
	from  pzhang.g_val_fcst f, g_item   i
		where f.item_nbr = i.item_nbr
	group by 1 ,2  
	union all 
	sel family,  salesdate, sum( ln(fcst + 1 )) as fcst  
	from pzhang.g_test_fcst_T007 f,
		pzhang.g_test_2 t, 
		g_item   i
	where t.item_nbr = i.item_nbr
	and f.id = t.id 
	group by 1 ,2  
) f
on s.family  = f.family
and s.salesdate = f.salesdate 
order by 1,2
#### Load Store Data
DROP TABLE pzhang.g_store; 
CREATE SET TABLE pzhang.g_store
     (
	  store_nbr SMALLINT,
	  city VARCHAR(60),
	  state VARCHAR(60),
      storetype char(1),
      storecluster BYTEINT)
PRIMARY INDEX ( store_nbr );

###### Check high sales on Aug 1st

sel  itemclass, salesdate, sum(unit_sales) 
from pzhang.g_train t , g_item i
where salesdate between '2017-07-30' and '2017-08-02'
and t.item_nbr = i.item_nbr 
and family = 'GROCERY I '
group by 1,2  order by 1,2 
;


-- DROP TABLE pzhang.g_store; 
CREATE MULTISET TABLE pzhang.g_transaction(
	  salesdate DATE,
	  store_nbr	 BYTEINT,
      transactions INTEGER )
PRIMARY INDEX ( store_nbr );

sel  salesdate, sum(transactions) 
from pzhang.g_transaction group by 1 order by 1 



##########################################################################
####################expand table, calc uplift ############################
##########################################################################



DROP TABLE pzhang.g_expand ; 

CREATE MULTISET TABLE pzhang.g_expand
     (
      store_nbr SMALLINT,
      item_nbr INTEGER,
      salesdate DATE FORMAT 'YYYY-MM-DD',
      sales DECIMAL(18,3),
      onpromotion BYTEINT)
PRIMARY INDEX ( store_nbr ,item_nbr );

 insert into pzhang.g_expand
sel a.store_nbr,  a.item_nbr, a.salesdate, 
coalesce(f.sales,0) as sales, 
coalesce(f.onpromotion,0) as onpromotion
from 
(
	sel store_nbr,  item_nbr, c.salesdate 
	from 
	( sel item_nbr,  store_nbr , min(salesdate) as min_date, max(salesdate) as max_date from pzhang.g_full group by 1 ,2 ) b
	, ( sel salesdate  from pzhang.g_calendar  ) c
	where c.salesdate between min_date and max_date 
) a
left join pzhang.g_full f
on a.item_nbr = f.item_nbr
and a.store_nbr = f.store_nbr
and a.salesdate =f.salesdate 
;



sel 
a.store_nbr, a.item_nbr, avg_sales_reg, avg_sales_promo, avg_sales_promo/avg_sales_reg as uplift
from 
(
	 sel store_nbr, item_nbr, average(ln(sales+1)) as avg_sales_reg  from pzhang.g_expand 
	 where salesdate between '2017-01-01' and '2017-05-31' 
	 and onpromotion = 0 group by 1 ,2 
	 having avg_sales_reg > 0 
 ) a
 inner join 
 (
	 sel store_nbr, item_nbr, average(ln(sales+1)) as avg_sales_promo  from pzhang.g_expand 
	 where salesdate between '2017-01-01' and '2017-05-31' 
	 and onpromotion = 1 group by 1 ,2 
 ) b
on a.item_nbr = b.item_nbr
and a.store_nbr = b.store_nbr
;

##############################################################################
############## Set forecast to zero on the missing record for Wed Promo ######
##############################################################################


replace view g_wed_promo
AS 	
sel s.store_nbr, s.item_nbr , 
	count(*) as wed_cnt,  sum(onpromotion) as sum_promo
from  pzhang.g_train  s
inner join pzhang.g_calendar c
on s.salesdate = c.salesdate
where  s.salesdate >= '2017-01-02'
	and daynumber  = 1 
	having wed_cnt = sum_promo
	group by 1 ,2 
;


	sel id , store_nbr,  item_nbr, salesdate    from pzhang.g_test where
	 (store_nbr,  item_nbr)
	in ( select store_nbr,  item_nbr from g_wed_promo) 
	and salesdate in ( '2017-08-16', '2017-08-23', '2017-08-30' ) 
	and onpromotion = 0 
;


##############################################################################
CREATE TABLE pzhang. g_continue_promo_s_i
as ( 
	sel 
	item_nbr, store_nbr, min(salesdate) as startdate, max(salesdate) as enddate 
	from  pzhang.g_test
	where (item_nbr, store_nbr) in (
		sel a.item_nbr, a.store_nbr
		from 
		(
			sel  item_nbr, store_nbr, sum(onpromotion) as sum_promo 	 
			from  pzhang.g_test s
			inner join pzhang.g_calendar c
			on s.salesdate = c.salesdate
			where s.salesdate > '2017-08-15' 
			having sum_promo between 11 and 15 
			group by 1, 2 
		) a , 
		(
			sel  item_nbr, store_nbr, sum(onpromotion) as sum_promo 	 
			from  pzhang.g_test s
			inner join pzhang.g_calendar c
			on s.salesdate = c.salesdate
			where s.salesdate between '2017-08-16'  and '2017-08-18'
			having sum_promo >=  2 
			group by 1, 2 
		) b 
		where a.store_nbr = b.store_nbr
		and a.item_nbr = b.item_nbr 
	)  
	and onpromotion = 1 
	group by 1,2 
) WITH DATA 
primary index (	item_nbr, store_nbr )  
;

SELECT t.id, t. store_nbr, t. item_nbr, salesdate   FROM  pzhang.g_test t
INNER JOIN pzhang. g_continue_promo_s_i p
ON t.store_nbr = p.store_nbr
		and t.item_nbr = p.item_nbr 
		and t.salesdate between p.startdate and p.enddate 
ORDER BY t. store_nbr, t. item_nbr, salesdate 

WHERE t.onpromotion = 0 