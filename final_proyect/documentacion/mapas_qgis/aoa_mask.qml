<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry min="0" max="0" percentTransparent="100"/>
        </singleValuePixelList>
      </rasterTransparency>
      <colorPalette>
        <paletteEntry color="#000000" label="NoData (Océano / Exterior)" alpha="0" value="0"/>
        <paletteEntry color="#1b9e77" label="1 = Dentro del AOA (DI &lt;= 2.51, Interpolación Admisible)" alpha="255" value="1"/>
        <paletteEntry color="#d95f02" label="2 = Fuera del AOA (DI &gt; 2.51, Extrapolación No Confiable)" alpha="255" value="2"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeRed="255" colorizeOn="0" colorizeStrength="100" colorizeBlue="128" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>