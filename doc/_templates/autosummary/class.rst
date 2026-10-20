{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :template: autosummary/method.rst
   {% for item in methods %}
       {% if item != "__init__" %}
          ~{{ name }}.{{ item }}
       {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
