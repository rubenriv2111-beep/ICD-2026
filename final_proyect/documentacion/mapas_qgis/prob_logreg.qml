<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" opacity="1" alphaBand="-1" band="1" classificationMin="0" classificationMax="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0</cumulativeCutLower>
        <cumulativeCutUpper>1</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0" classificationMode="1" labelPrecision="2">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" name="color1" value="68,1,84,255"/>
              <Option type="QString" name="color2" value="253,231,37,255"/>
              <Option type="QString" name="stops" value="0.25;59,82,139,255:0.5;33,145,140,255:0.75;94,201,98,255"/>
            </Option>
          </colorramp>
          <item color="#440154" label="0.00 (Baja Susceptibilidad)" value="0" alpha="255"/>
          <item color="#3b528b" label="0.25" value="0.25" alpha="255"/>
          <item color="#21918c" label="0.50 (Indiferente)" value="0.5" alpha="255"/>
          <item color="#5ec962" label="0.75" value="0.75" alpha="255"/>
          <item color="#fde725" label="1.00 (Alta Susceptibilidad)" value="1" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeRed="255" colorizeOn="0" colorizeStrength="100" colorizeBlue="128" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>