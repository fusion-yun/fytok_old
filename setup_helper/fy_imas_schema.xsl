<?xml version="1.0" encoding="UTF-8"?>
<!--  
  Generate Python class (FyTok IDS) from IDSDef.xml file 
  
  copyright:
     @ASIPP, 2023,

  authors:
     Zhi YU, @ASIPP

  changes:
    2023-04-26: 0.0.1, ZY, initial from fy_imas.xsl
     

-->
<xsl:stylesheet  
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions"	
  xmlns:my="http://www.example.com/my"  
  xmlns:saxon="http://saxon.sf.net/"
  version="3.0"
>
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="no"/>
<!-- <xsl:strip-space elements="*"/> -->
<xsl:include href="fy_imas_type_convert.xsl"/>
<xsl:include href="fy_imas_utils.xsl"/>

<xsl:param name="FY_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="DD_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="DD_WRAPPER_DIR"  as="xs:string"  select="translate($DD_GIT_DESCRIBE,'.-','__')" />

<!-- <xsl:param name="LICENSE_ANNOTATION" select=" N/A " /> -->

<xsl:param name="line-width" select="80" />
   
<xsl:param name="DD_BASE_DIR" as="xs:string" required='true' />

<xsl:variable name="LICENSE_ANNOTATION" ># Generate by FyTok (rev: <xsl:value-of select="$FY_GIT_DESCRIBE"/>): builder/fy_imas_schema.xsl at <xsl:value-of  select="current-dateTime()" /> </xsl:variable>

<!-- Directory:  _imas  -->
<xsl:template match="/*">  

  <!-- <xsl:apply-templates select="xs:element[@name='physics_data_dictionary']" mode="file_init_py" /> -->

  <!-- Scan for all constant identify ENUM -->
  <xsl:variable name="constants_list"   select="for $f in xs:include  return (document(concat($DD_BASE_DIR,$f/@schemaLocation))//doc_identifier ) " />
  <xsl:variable name="constants_list"   select="for $f in $constants_list  return  if (starts-with($f,'utilities/')) then $f else () " />
  
  <xsl:call-template name="file_utilities">    
    <xsl:with-param name="constants_list" select="$constants_list" />
  </xsl:call-template>

  <xsl:for-each select="xs:include[@schemaLocation!='utilities/dd_support.xsd']">
      <xsl:apply-templates select="document(concat($DD_BASE_DIR,./@schemaLocation))/*" mode="single_ids" />   
  </xsl:for-each>
  <xsl:value-of select="$DD_GIT_DESCRIBE"/>
</xsl:template>


<!-- FILE:  __init__.py -->
<xsl:template match="xs:element[@name='physics_data_dictionary']" mode="file_init_py">
  <xsl:result-document method="text" href="__init__.py">#<xsl:copy-of select="$LICENSE_ANNOTATION" />
#<xsl:value-of select="xs:annotation/xs:documentation"/>
#
#  From IMAS/dd (<xsl:value-of select="$DD_GIT_DESCRIBE"/>)
#  

    #__path__ = __import__('pkgutil').extend_path(__path__, __name__)

    #__fy_rev__  ="<xsl:value-of select="$FY_GIT_DESCRIBE"/>"
    #__version__ ="<xsl:value-of select="$DD_GIT_DESCRIBE"/>"
    #__cocos__   ="<xsl:value-of select="xs:annotation/xs:appinfo/cocos"/>"

    <xsl:for-each select="xs:complexType/xs:sequence/xs:element">
    # from .<xsl:value-of select="@ref"/>  import _T_<xsl:value-of select="@ref"/> 
    </xsl:for-each>

    </xsl:result-document>
  </xsl:template>



<!-- FILE:  utilities.yaml -->
<xsl:template name="file_utilities">
  <xsl:param name="constants_list"/>
  <xsl:variable name="root" select="document(concat($DD_BASE_DIR,'utilities/dd_support.xsd'))/*"/>
  <xsl:result-document method="text" href='utilities.py'>
    <xsl:copy-of select="$LICENSE_ANNOTATION" />   
    <xsl:text># This module containes the _FyTok_ wrapper of IMAS/dd/utilities.py </xsl:text>
    <xsl:text>&#xA;</xsl:text>

    <xsl:for-each select="$constants_list"> 
      <xsl:apply-templates  select = "document(concat($DD_BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
    </xsl:for-each>

    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=0] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=1] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=2] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=3] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=4] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=5] " mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=6] " mode="DEFINE"/>

    <xsl:apply-templates select="$root/xs:element" mode="DEFINE"/>
    <xsl:value-of select="unparsed-text('fy_imas_utilities.py')"/>
  </xsl:result-document>   
</xsl:template>

<!-- FILE:  {@name}.py -->
<xsl:template match = "xs:schema" mode = "single_ids"> 
  <xsl:variable name="filename" select="xs:element/@name"/>
  <!-- <xsl:message> DEBUG: create <xsl:value-of select="$filename"/>.py </xsl:message> -->
  <xsl:result-document method="text" href="{$filename}.yaml"  >
    <xsl:copy-of select="$LICENSE_ANNOTATION" />   
    <xsl:text>&#xA;#  This file containes the JSON Schema of IMAS/dd/</xsl:text><xsl:value-of select="xs:element/@name" />  
    <xsl:text>&#xA;</xsl:text>

    <xsl:variable name="cls_list" select="for $k in //doc_identifier return if (not(starts-with($k,'utilities/'))) then   $k  else ()"/>
    <xsl:for-each select="distinct-values($cls_list)">
      <xsl:apply-templates  select = "document(concat($DD_BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
    </xsl:for-each>

    <xsl:variable name="root" select="." />
    
    <xsl:apply-templates select="xs:complexType" mode="DEFINE"/> 
        
    <xsl:apply-templates select="xs:element" mode="DEFINE_ELEMENT_AS_IDS"/>

  </xsl:result-document>     
</xsl:template>

<!-- as_python_value 将 节点内容转换为 python dict/list/string
  - 如果节点是一个简单类型，则它将其转换为一个字符串。
  - 如果节点是一个复杂类型，则它将其转换为一个字典，其中包含所有子节点的值。
  - 如果有多个同名兄弟节点，则它将它们存储在一个列表中；否则，它将它们存储为单个值。   
-->

<xsl:template match="*" mode="as_python_kw">
  <xsl:if test="not(preceding-sibling::*[name() = name(current())])">
    <xsl:value-of select="my:quote(name())" />>: 
    <xsl:variable name="siblings" select="../child::*[name() = name(current())]" />
    <xsl:choose>
      <xsl:when test="count($siblings) > 1">
        <xsl:text>[</xsl:text>
        <xsl:apply-templates select="$siblings" mode="as_python_value"/>
        <xsl:text>]</xsl:text>
      </xsl:when>
      <xsl:otherwise><xsl:value-of select="my:quote($siblings)" /></xsl:otherwise>
    </xsl:choose>
    <xsl:if test="position() != last()">,</xsl:if>
  </xsl:if>
</xsl:template>

<xsl:template match="*" mode="as_python_value">
  <xsl:choose>
    <xsl:when test="./*">
      <xsl:text>{</xsl:text><xsl:apply-templates select="*" mode="as_python_kw" /><xsl:text>}</xsl:text>
    </xsl:when>
    <xsl:otherwise><xsl:value-of select="my:quote(.)" /></xsl:otherwise>
  </xsl:choose>
  <xsl:if test="position() != last()">,</xsl:if>
</xsl:template>

<!-- as_python_kwargs 将子节点转换为 key=value 形式，可用作 python 函数 kwargs 参数  
  - 如果有多个同名兄弟节点，则它将它们存储在一个列表中；否则，它将它们存储为单个值。
-->

<xsl:template match="*" mode="as_python_kwargs_one">  
  <xsl:if test="not(preceding-sibling::*[name() = name(current())])">
    <xsl:value-of select="my:py_keyword(name())" /><xsl:text>=</xsl:text>
    <xsl:variable name="siblings" select="../child::*[name() = name(current())]" />
    <xsl:choose>
      <xsl:when test="count($siblings) > 1"><xsl:text>[</xsl:text><xsl:apply-templates select="$siblings" mode="as_python_value"/><xsl:text>]</xsl:text></xsl:when>
      <xsl:otherwise><xsl:apply-templates select="$siblings"  mode="as_python_value" /> </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="position() != last()">,</xsl:if>
  </xsl:if>  
</xsl:template>

<xsl:template match="*" mode="as_python_kwargs">
  <xsl:choose>
    <xsl:when test="./*">
      <xsl:apply-templates select="*" mode="as_python_kwargs_one" />
    </xsl:when>
    <xsl:otherwise></xsl:otherwise>
  </xsl:choose>
  
</xsl:template>

<!--  ######################################################################################### -->


<xsl:template match = "xs:appinfo">
  <xsl:param name="indent" as="xs:integer" select="0"/>
  <xsl:for-each select="./*">
    <xsl:value-of select="my:indent($indent)"/><xsl:value-of select="name()" />: <xsl:value-of select="." /><xsl:text>&#xA;</xsl:text>
  </xsl:for-each>
</xsl:template>

<xsl:template match = "xs:annotation">
  <xsl:param name="indent" as="xs:integer" select="0"/>
  <xsl:value-of select="my:indent($indent)"/>annotation:<xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>document: "<xsl:value-of select="my:line-wrap(xs:documentation, $line-width, $indent*2+1)"/>"<xsl:text>&#xA;</xsl:text>
  <!-- <xsl:apply-templates select="xs:appinfo"> <xsl:with-param name="indent" select="$indent+1" /> </xsl:apply-templates> -->
  <xsl:for-each select="xs:appinfo/*">
    <xsl:value-of select="my:indent($indent+1)"/><xsl:value-of select="name()" />: <xsl:value-of select="." /><xsl:text>&#xA;</xsl:text>
  </xsl:for-each>
</xsl:template>


 

<xsl:template match="xs:element[@name  and  not(xs:annotation/xs:appinfo/lifecycle_status='obsolescent')]" mode="DECLARE">
  <xsl:param name="indent" as="xs:integer" select="0"/>

  <xsl:variable name="prop_name">
    <xsl:choose>
      <xsl:when test="@ref"><xsl:value-of select="my:py_keyword(@ref)"/></xsl:when>
      <xsl:otherwise><xsl:value-of select="my:py_keyword(@name)"/></xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="type_hint">
    <xsl:choose>
      <xsl:when test="@ref"><xsl:value-of select="@ref"/></xsl:when>
      <xsl:when test="@type"><xsl:value-of select="@type"/></xsl:when>
      <xsl:otherwise><xsl:value-of select="xs:complexType/xs:group/@ref"/></xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="type_hint">
    <xsl:choose>
      <xsl:when test="($type_hint='INT_1D' or $type_hint='int_1d_type') and normalize-space(xs:annotation/xs:appinfo/coordinate1)='1...N' ">List[int]</xsl:when>          
      <xsl:when test="($type_hint='FLT_1D' or $type_hint='flt_1d_type') and ends-with(xs:annotation/xs:appinfo/coordinate1,'time')">Function[float]</xsl:when>
      <xsl:when test="($type_hint='FLT_1D' or $type_hint='flt_1d_type') and ends-with(xs:annotation/xs:appinfo/coordinate1,'rho_tor_norm')">Function[float]</xsl:when>                  
      <xsl:when test="($type_hint='FLT_1D' or $type_hint='flt_1d_type') and ends-with(xs:annotation/xs:appinfo/coordinate1,'psi')">Function[float]</xsl:when>                     
      <xsl:when test="($type_hint='FLT_2D' or $type_hint='flt_2d_type') and normalize-space(xs:annotation/xs:appinfo/coordinate1)=('../grid/dim1')  and normalize-space(xs:annotation/xs:appinfo/coordinate2)=('../grid/dim2')">Field[float]</xsl:when>          
      <xsl:when test="$type_map/entry[@key=$type_hint]"><xsl:value-of select="$type_map/entry[@key=$type_hint]"/></xsl:when>          
      <xsl:when test="xs:annotation/xs:appinfo/doc_identifier">_E_<xsl:value-of select = "document(concat($DD_BASE_DIR, xs:annotation/xs:appinfo/doc_identifier))/constants/@name"/></xsl:when>
      <xsl:otherwise><xsl:value-of select="$type_hint"/> </xsl:otherwise>   
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="type_hint">
    <xsl:choose>
      <xsl:when test="@maxOccurs">
        <xsl:choose>
          <xsl:when test="ends-with(xs:annotation/xs:appinfo/coordinate1,'time')">TimeSeriesAoS[<xsl:value-of select="$type_hint" />]</xsl:when>      
          <xsl:otherwise>AoS[<xsl:value-of select="$type_hint" />]</xsl:otherwise>               
        </xsl:choose>
      </xsl:when>   
      <xsl:otherwise><xsl:value-of select="$type_hint"/> </xsl:otherwise>   
    </xsl:choose>
  </xsl:variable>
  <xsl:value-of select="my:indent($indent)"/><xsl:value-of select="$prop_name"/>:  <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>type: <xsl:value-of select="$type_hint" /><xsl:text>&#xA;</xsl:text>
  <xsl:apply-templates select="xs:annotation"> <xsl:with-param name="indent" select="$indent+1"/>  </xsl:apply-templates>
  <xsl:text>&#xA;</xsl:text>
</xsl:template>
 
<xsl:template match="xs:element[xs:annotation/xs:appinfo/lifecycle_status='obsolescent']" mode="DECLARE" />
<!-- <xsl:template match="xs:sequence" mode="property_list">
  <xsl:for-each select="xs:element[@name!='code' and @name!='time' and @name!='ids_properties' ]">
    <xsl:choose>
      <xsl:when test = "@ref" >
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@ref)"/>   : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo"  mode="as_python_kwargs"/>)
      </xsl:when>
      <xsl:when test= "not(@ref) and not(xs:annotation/xs:appinfo/doc_identifier)">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/>  : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo"  mode="as_python_kwargs"/>)
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
      </xsl:when>
      <xsl:when test = "not(@ref) and (xs:annotation/xs:appinfo/doc_identifier)">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/> : _E_<xsl:value-of select = "document(concat($DD_BASE_DIR, xs:annotation/xs:appinfo/doc_identifier))/constants/@name"  />  =  sp_property()
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
      </xsl:when>      
    </xsl:choose>
  </xsl:for-each>  
</xsl:template> -->
<!-- <xsl:template match = "xs:complexType[xs:sequence/xs:element[@name='code']]" mode = "DEFINE"> 
      <xsl:text>&#xA;&#xA;&#x9;</xsl:text><xsl:value-of select="@name" />: <xsl:text>&#xA;</xsl:text>
      <xsl:text>&#x9;&#x9;</xsl:text>annotation: <xsl:text>&#xA;</xsl:text>
          <xsl:apply-templates select="xs:annotation"> <xsl:with-param name="indent" select="$indent+1"/>  </xsl:apply-templates>
          <xsl:apply-templates select="xs:sequence/xs:element[@name!='code']" mode="DECLARE" />
</xsl:template> -->
<xsl:template match = "xs:complexType[not($type_map/entry[@key=@name])]" mode = "DEFINE"> 
  <xsl:param name="indent" as="xs:integer" select="0"/>

  <xsl:value-of select="my:indent($indent)"/><xsl:value-of select="@name" />: <xsl:text>&#xA;</xsl:text>

  <xsl:apply-templates select="xs:annotation"> <xsl:with-param name="indent" select="$indent+1"/>  </xsl:apply-templates>

  <xsl:text>&#xA;</xsl:text>

  <xsl:value-of select="my:indent($indent+1)"/>properties: <xsl:text>&#xA;</xsl:text>
    <xsl:apply-templates select="xs:sequence/xs:element" mode="DECLARE"> <xsl:with-param name="indent" select="$indent+2"/>  </xsl:apply-templates>

</xsl:template>

<xsl:template match = "constants[@identifier='yes']" mode = "CONSTANTS_IDENTIFY"> 
  <xsl:param name="indent" as="xs:integer" select="0"/>
  <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent)"/><xsl:value-of select="@name"/>:<xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>#<xsl:value-of select="my:line-wrap(header, $line-width, 2)"/><xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/># xpath: <xsl:value-of select="dd_instance/@xpath"/><xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>#"<xsl:text>&#xA;</xsl:text>
  <xsl:for-each select="int">
      <xsl:value-of select="my:indent($indent+1)"/><xsl:value-of select="my:py_keyword(@name)"/> : <xsl:value-of select="."/>  # <xsl:value-of select="@description"/> <xsl:text>&#xA;</xsl:text>
  </xsl:for-each>
  <xsl:text>&#xA;</xsl:text>
</xsl:template>

<xsl:template match = "xs:element[not($type_map/entry[@key=@name])]" mode = "DEFINE"> 
  <xsl:param name="indent" as="xs:integer" select="0"/>
  <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>type: <xsl:value-of select="@name" /><xsl:text>&#xA;</xsl:text>
  <xsl:apply-templates select="xs:annotation"> <xsl:with-param name="indent" select="$indent+1"/>  </xsl:apply-templates>
   
  <xsl:apply-templates select="xs:complexType/xs:sequence/xs:element" mode="DECLARE" />
  <xsl:text>&#xA;</xsl:text>
</xsl:template>

<xsl:template match = "xs:element" mode = "DEFINE_ELEMENT_AS_IDS"> 
  <xsl:param name="indent" as="xs:integer" select="0"/>
  <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent)"/><xsl:value-of select="@name" />: <xsl:text>&#xA;</xsl:text>

  <xsl:apply-templates select="xs:annotation"><xsl:with-param name="indent" select="$indent+1"/></xsl:apply-templates> 
  <xsl:value-of select="my:indent($indent+2)"/>dd_version: "<xsl:value-of select="$DD_GIT_DESCRIBE" />" <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+2)"/>ids_name: "<xsl:value-of select="@name" />"<xsl:text>&#xA;</xsl:text>
  
  <xsl:text>&#xA;</xsl:text>
  <xsl:value-of select="my:indent($indent+1)"/>properties:<xsl:text>&#xA;</xsl:text>
  <xsl:apply-templates select="xs:complexType/xs:sequence/xs:element" mode="DECLARE" > <xsl:with-param name="indent" select="$indent+2"/> </xsl:apply-templates>
  <xsl:text>&#xA;</xsl:text>
</xsl:template>

</xsl:stylesheet>

