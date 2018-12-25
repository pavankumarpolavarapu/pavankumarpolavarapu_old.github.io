---
layout: page
title: Notebook
description: Pavans notebook
permalink: /notebook/
---

Contains my thoughts and work on the things that I wish to keep track. Please feel free to email me at pavan.vnr@gmail.com if you find any mistakes / errors.

<ul>
  {% for post in site.categories.notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>